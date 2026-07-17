from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TenantSettings:
    tenant_id: str
    path_prefix: str
    env_prefix: str
    backend_dir: Path
    chatbot_module: str
    subgraph_module: str
    deck_module: str
    ppt_template_env: str
    ppt_logo_env: str
    daily_pulse_defaults: tuple[str, ...]


TENANT_SETTINGS: dict[str, TenantSettings] = {
    "geron": TenantSettings(
        tenant_id="geron",
        path_prefix="/geron",
        env_prefix="GERON",
        backend_dir=BACKEND_ROOT / "Geron_Backend",
        chatbot_module="chatbot8",
        subgraph_module="subgraph_14",
        deck_module="deck_creator_agent_7",
        ppt_template_env="GERON_PPT_TEMPLATE_PATH",
        ppt_logo_env="GERON_PPT_LOGO_PATH",
        daily_pulse_defaults=(
            "Are we seeing strong short-term sales momentum?",
            "How are we doing in terms of adding new businesses?",
        ),
    ),
    "crinetics": TenantSettings(
        tenant_id="crinetics",
        path_prefix="/crinetics",
        env_prefix="CRINETICS",
        backend_dir=BACKEND_ROOT / "Crinetics_Backend",
        chatbot_module="chatbot_Crinetics",
        subgraph_module="subgraph_Crinetics",
        deck_module="deck_creator_agent_Crinetics",
        ppt_template_env="CRINETICS_PPT_TEMPLATE_PATH",
        ppt_logo_env="CRINETICS_PPT_LOGO_PATH",
        daily_pulse_defaults=(
            "Give me the total number of enrollments.",
            "Give me the total number of dispenses.",
        ),
    ),
}

ALLOWED_TENANTS = tuple(TENANT_SETTINGS.keys())

GENERIC_ENV_NAMES: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "DB_URI",
    "TEST_DB_URI_FALLBACK",
    "POSTGRES_URL",
    "POSTGRES_URI",
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_ANALYTICS_SCHEMA",
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
    "PPT_TEMPLATE_PATH",
    "PPT_LOGO_PATH",
)


def get_tenant_settings(tenant_id: str) -> TenantSettings:
    return TENANT_SETTINGS[tenant_id]


def tenant_env_name(tenant_id: str, generic_name: str) -> str:
    return f"{TENANT_SETTINGS[tenant_id].env_prefix}_{generic_name}"
