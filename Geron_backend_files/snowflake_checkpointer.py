import json
import pickle
import base64
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple
from contextlib import contextmanager

import snowflake.connector
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class SnowflakeSaver(BaseCheckpointSaver):
    """
    LangGraph checkpoint saver backed by Snowflake.
    Drop-in replacement for SqliteSaver.
    """

    serde = JsonPlusSerializer()

    def __init__(self, conn: snowflake.connector.SnowflakeConnection):
        super().__init__()
        self.conn = conn
        self._setup()

    def _setup(self):
        """Create checkpoint tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS LANGGRAPH_CHECKPOINTS (
                thread_id        VARCHAR      NOT NULL,
                checkpoint_ns    VARCHAR      NOT NULL DEFAULT '',
                checkpoint_id    VARCHAR      NOT NULL,
                parent_id        VARCHAR,
                checkpoint       VARIANT,
                metadata         VARIANT,
                created_at       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS LANGGRAPH_CHECKPOINT_WRITES (
                thread_id        VARCHAR      NOT NULL,
                checkpoint_ns    VARCHAR      NOT NULL DEFAULT '',
                checkpoint_id    VARCHAR      NOT NULL,
                task_id          VARCHAR      NOT NULL,
                idx              INTEGER      NOT NULL,
                channel          VARCHAR      NOT NULL,
                type             VARCHAR,
                value            VARIANT,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
        """)
        cursor.close()

    def get_tuple(self, config: Dict) -> Optional[CheckpointTuple]:
        """Fetch the latest (or a specific) checkpoint for a thread."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        cursor = self.conn.cursor()
        try:
            if checkpoint_id:
                cursor.execute(
                    """
                    SELECT thread_id, checkpoint_ns, checkpoint_id,
                           parent_id, checkpoint, metadata
                    FROM LANGGRAPH_CHECKPOINTS
                    WHERE thread_id = %s
                      AND checkpoint_ns = %s
                      AND checkpoint_id = %s
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT thread_id, checkpoint_ns, checkpoint_id,
                           parent_id, checkpoint, metadata
                    FROM LANGGRAPH_CHECKPOINTS
                    WHERE thread_id = %s
                      AND checkpoint_ns = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                )

            row = cursor.fetchone()
            if row is None:
                return None

            tid, ns, cid, parent_id, checkpoint_json, metadata_json = row

            # Snowflake returns VARIANT as already-parsed Python objects
            checkpoint = self.serde.loads_typed(
                ("json", json.dumps(checkpoint_json))
            )
            metadata = self.serde.loads_typed(
                ("json", json.dumps(metadata_json))
            ) if metadata_json else {}

            # Fetch pending writes
            cursor.execute(
                """
                SELECT task_id, channel, type, value
                FROM LANGGRAPH_CHECKPOINT_WRITES
                WHERE thread_id = %s
                  AND checkpoint_ns = %s
                  AND checkpoint_id = %s
                ORDER BY idx
                """,
                (tid, ns, cid),
            )
            pending_writes = []
            for task_id, channel, w_type, w_value in cursor.fetchall():
                pending_writes.append((
                    task_id,
                    channel,
                    self.serde.loads_typed((w_type, json.dumps(w_value))),
                ))

            config_out = {
                "configurable": {
                    "thread_id": tid,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cid,
                }
            }
            parent_config = (
                {
                    "configurable": {
                        "thread_id": tid,
                        "checkpoint_ns": ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            )

            return CheckpointTuple(
                config=config_out,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )
        finally:
            cursor.close()

    def list(
        self,
        config: Dict,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread, newest first."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        query = """
            SELECT thread_id, checkpoint_ns, checkpoint_id,
                   parent_id, checkpoint, metadata
            FROM LANGGRAPH_CHECKPOINTS
            WHERE thread_id = %s AND checkpoint_ns = %s
        """
        params = [thread_id, checkpoint_ns]

        if before:
            query += " AND checkpoint_id < %s"
            params.append(get_checkpoint_id(before))

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {int(limit)}"

        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            for tid, ns, cid, parent_id, checkpoint_json, metadata_json in cursor.fetchall():
                checkpoint = self.serde.loads_typed(
                    ("json", json.dumps(checkpoint_json))
                )
                metadata = self.serde.loads_typed(
                    ("json", json.dumps(metadata_json))
                ) if metadata_json else {}

                config_out = {
                    "configurable": {
                        "thread_id": tid,
                        "checkpoint_ns": ns,
                        "checkpoint_id": cid,
                    }
                }
                parent_config = (
                    {
                        "configurable": {
                            "thread_id": tid,
                            "checkpoint_ns": ns,
                            "checkpoint_id": parent_id,
                        }
                    }
                    if parent_id
                    else None
                )
                yield CheckpointTuple(
                    config=config_out,
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                    pending_writes=[],
                )
        finally:
            cursor.close()

    def put(
        self,
        config: Dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> Dict:
        """Persist a checkpoint to Snowflake."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        type_, serialized_metadata = self.serde.dumps_typed(metadata)

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                MERGE INTO LANGGRAPH_CHECKPOINTS AS target
                USING (
                    SELECT %s AS thread_id,
                           %s AS checkpoint_ns,
                           %s AS checkpoint_id,
                           %s AS parent_id,
                           PARSE_JSON(%s) AS checkpoint,
                           PARSE_JSON(%s) AS metadata
                ) AS source
                ON  target.thread_id     = source.thread_id
                AND target.checkpoint_ns = source.checkpoint_ns
                AND target.checkpoint_id = source.checkpoint_id
                WHEN MATCHED THEN UPDATE SET
                    parent_id  = source.parent_id,
                    checkpoint = source.checkpoint,
                    metadata   = source.metadata
                WHEN NOT MATCHED THEN INSERT
                    (thread_id, checkpoint_ns, checkpoint_id, parent_id, checkpoint, metadata)
                VALUES
                    (source.thread_id, source.checkpoint_ns, source.checkpoint_id,
                     source.parent_id, source.checkpoint, source.metadata)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_id,
                    serialized_checkpoint,
                    serialized_metadata,
                ),
            )
        finally:
            cursor.close()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes for a checkpoint step."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        cursor = self.conn.cursor()
        try:
            for idx, (channel, value) in enumerate(writes):
                type_, serialized_value = self.serde.dumps_typed(value)
                cursor.execute(
                    """
                    MERGE INTO LANGGRAPH_CHECKPOINT_WRITES AS target
                    USING (
                        SELECT %s AS thread_id,
                               %s AS checkpoint_ns,
                               %s AS checkpoint_id,
                               %s AS task_id,
                               %s AS idx,
                               %s AS channel,
                               %s AS type,
                               PARSE_JSON(%s) AS value
                    ) AS source
                    ON  target.thread_id     = source.thread_id
                    AND target.checkpoint_ns = source.checkpoint_ns
                    AND target.checkpoint_id = source.checkpoint_id
                    AND target.task_id       = source.task_id
                    AND target.idx           = source.idx
                    WHEN MATCHED THEN UPDATE SET
                        channel = source.channel,
                        type    = source.type,
                        value   = source.value
                    WHEN NOT MATCHED THEN INSERT
                        (thread_id, checkpoint_ns, checkpoint_id, task_id,
                         idx, channel, type, value)
                    VALUES
                        (source.thread_id, source.checkpoint_ns, source.checkpoint_id,
                         source.task_id, source.idx, source.channel,
                         source.type, source.value)
                    """,
                    (
                        thread_id, checkpoint_ns, checkpoint_id,
                        task_id, idx, channel, type_, serialized_value,
                    ),
                )
        finally:
            cursor.close()