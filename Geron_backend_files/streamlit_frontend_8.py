import streamlit as st
from chatbot7 import build_chatbot
from langchain_core.messages import HumanMessage, AIMessage
#from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from datetime import datetime, UTC, timedelta
import matplotlib.pyplot as plt
import uuid
import sqlite3
import plotly.express as px
import re
from typing import Dict, Optional
import pandas as pd
import time
from threading import Thread

####################################################################### UI STYLE ################################################################################
st.set_page_config(
    page_title="A360",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

[data-testid="stChatMessage"] {
    padding: 14px 18px;
    border-radius: 14px;
    margin-bottom: 10px;
    line-height: 1.6;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 44px;
    font-weight: 500;
}

section[data-testid="stSidebar"] {
    background-color: #fafafa;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Remove sidebar button styling */
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    text-align: left !important;
    padding: 0px 0px !important;
}

/* Hover like navigation item */
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background-color: rgba(0,0,0,0.05) !important;
}

/* Active click effect */
section[data-testid="stSidebar"] button[kind="secondary"]:focus {
    outline: none !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* reduce vertical spacing between sidebar elements */
section[data-testid="stSidebar"] div.stButton {
    margin-bottom: 2px;
}

/* reduce button height */
section[data-testid="stSidebar"] button {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
    min-height: 32px !important;
    line-height: 1.2 !important;
}

/* remove extra block spacing Streamlit adds */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
    gap: 0.2rem !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Remove vertical spacing between ALL sidebar elements */
section[data-testid="stSidebar"] div[data-testid="element-container"] {
    margin-bottom: 0rem !important;
    padding-bottom: 0rem !important;
}

/* Remove block spacing */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0rem !important;
}

/* Ultra compact buttons */
section[data-testid="stSidebar"] .stButton button {
    padding: 2px 6px !important;
    min-height: 26px !important;
    line-height: 1.1 !important;
    margin: 0 !important;
    font-size: 13px !important;
}

/* Remove default Streamlit invisible spacer */
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
    margin-bottom: 0rem !important;
}

/* Optional — make it ChatGPT tight */
section[data-testid="stSidebar"] div {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* ===== SIDEBAR TITLE ===== */
section[data-testid="stSidebar"] h1 {
    color: #0B1F3A !important;   /* Dark Blue */
    font-weight: 700 !important;
    letter-spacing: 0.2px;
}

/* ===== SIDEBAR SUBHEADERS ===== */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0B1F3A !important;   /* Same dark blue */
    font-weight: 600 !important;
}

/* keep normal text default */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: inherit !important;
}

</style>
""", unsafe_allow_html=True)


# st.markdown("""
#     <style>

#     /* ===== MAIN BACKGROUND ===== */
#     .stApp {
#         background-color: #F5F7FA;   /* soft grey page background */
#         color: #0F172A;
#     }

#     /* ===== SIDEBAR ===== */
#     section[data-testid="stSidebar"] {
#         background-color: #0B1F3A;   /* deep navy */
#         border-right: 1px solid rgba(255,255,255,0.08);
#     }

#     section[data-testid="stSidebar"] * {
#         color: #E5EEF8 !important;
#     }

#     /* Sidebar buttons */
#     section[data-testid="stSidebar"] .stButton > button {
#         background: transparent;
#         border: 1px solid rgba(255,255,255,0.25);
#         color: #E5EEF8;
#         border-radius: 8px;
#         padding: 10px 14px;
#         transition: all 0.25s ease;
#     }

#     section[data-testid="stSidebar"] .stButton > button:hover {
#         background-color: #1C3D6E;
#         border-color: #4FD1C5;   /* teal accent */
#         color: white;
#     }

#     /* ===== HEADINGS ===== */
#     h1, h2, h3, h4 {
#         color: #0B1F3A;
#         font-weight: 600;
#     }

#     /* ===== CHAT BUBBLES ===== */
#     [data-testid="stChatMessage"] {
#         background-color: white;
#         border-radius: 12px;
#         padding: 14px;
#         border: 1px solid #E2E8F0;
#         box-shadow: 0 2px 6px rgba(0,0,0,0.04);
#     }

#     /* User message */
#     [data-testid="stChatMessage"][aria-label="user"] {
#         background-color: #E6F7F5; /* soft teal tint */
#         border: 1px solid #4FD1C5;
#     }

#     /* ===== INPUT BOX ===== */
#     .stChatInput > div {
#         border-radius: 12px;
#         border: 1px solid #CBD5E1;
#     }

#     .stChatInput textarea {
#         background-color: white;
#     }

#     /* ===== PRIMARY BUTTONS ===== */
#     .stButton > button {
#         background-color: #0B1F3A;
#         color: white;
#         border-radius: 10px;
#         border: none;
#         padding: 10px 18px;
#         font-weight: 500;
#         transition: all 0.25s ease;
#     }

#     .stButton > button:hover {
#         background-color: #123D75;
#         transform: translateY(-1px);
#     }

#     /* ===== LINKS ===== */
#     a {
#         color: #0EA5A4 !important;   /* teal */
#         text-decoration: none;
#         font-weight: 500;
#     }

#     a:hover {
#         text-decoration: underline;
#     }

#     /* ===== DATAFRAMES ===== */
#     .stDataFrame {
#         border-radius: 12px;
#         border: 1px solid #E2E8F0;
#         overflow: hidden;
#     }

#     </style>
#     """, unsafe_allow_html=True)

####################################################################### UTILITY FUNCTIONS ################################################################################
def generate_thread_id():
    thread_id=datetime.now(UTC).isoformat(timespec="microseconds") + "T"
    return thread_id

def categorize_thread(thread_id: str):
    # remove trailing "T"
    ts = thread_id[:-1]

    thread_dt = datetime.fromisoformat(ts)
    now = datetime.now(UTC)

    today = now.date()
    thread_date = thread_dt.date()

    if thread_date == today:
        return "Today"

    elif thread_date == (today - timedelta(days=1)):
        return "Yesterday"

    elif thread_date >= (today - timedelta(days=7)):
        return "Last 7 Days"

    else:
        return "History"


def reset_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history']=[]
    st.session_state.app_mode = "home"



def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    state = st.session_state.chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in st.session_state.checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

def get_thread_title(messages, max_len=30):
    """
    Extracts a readable title from the first user message.
    """
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content.strip():
            title = msg.content.strip().replace("\n", " ")
            return title[:max_len] + ("..." if len(title) > max_len else "")
    return "Current conversation"

import re
from typing import Dict, Optional

def parse_agent_output(text: str) -> Dict[str, Optional[str]]:
    """
    Parses a combined agent output string into individual sections.

    Expected section headers:
    - SQL Query Executed:
    - Result Summary:
    - Query Results:
    - Visualization Code:
    - Relevant Questions:

    Returns a dict with keys:
    sql_query, result_summary, query_results, visualization_code, relevant_questions
    """

    sections = {
        "sql_query": None,
        "result_summary": None,
        "query_results": None,
        "visualization_code": None,
        "relevant_questions": None,
    }

    # Define headers once
    header_map = {
        "sql_query": "SQL Query Executed:",
        "result_summary": "Result Summary:",
        "query_results": "Query Results:",
        "visualization_code": "Visualization Code:",
        "relevant_questions": "Relevant Questions:",
    }

    # Create combined header regex safely
    all_headers_pattern = "|".join(re.escape(h) for h in header_map.values())

    # Build patterns dynamically
    patterns = {
        key: rf"{re.escape(header)}\s*(.*?)(?=\n(?:{all_headers_pattern})|$)"
        for key, header in header_map.items()
    }

    # Extract sections
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()

    return sections


def get_latest_assistant_text(messages):
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.additional_kwargs:
            return msg.content
    return None

def get_sql_results(messages):
    return [
        msg.additional_kwargs["data"]
        for msg in messages
        if msg.additional_kwargs.get("type") == "sql_result"
    ]

def get_visualizations(messages):
    return [
        msg.additional_kwargs["code"]
        for msg in messages
        if msg.additional_kwargs.get("type") == "visualization"
    ]

def get_latest_by_type(messages, msg_type):
    for msg in reversed(messages):
        if msg.additional_kwargs.get("type") == msg_type:
            return msg.additional_kwargs
    return None

def deserialize_df(serialized_df: dict) -> pd.DataFrame:
    if not serialized_df:
        return pd.DataFrame()
    return pd.DataFrame(
        data=serialized_df.get("data", []),
        columns=serialized_df.get("columns", [])
    )

def render_history():
    
    i = 0
    messages = st.session_state.message_history

    while i < len(messages):

        msg = messages[i]

        # ================= USER =================
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="assets/success.png"):
                st.markdown(
                    """
                    <div style="font-size:18px; font-weight:700; margin-bottom:6px;">
                        User:
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(msg.content)
            i += 1
            continue

        # ================= ASSISTANT GROUP =================
        if isinstance(msg, AIMessage) and not msg.additional_kwargs.get("type"):

            feedback_id = msg.additional_kwargs.setdefault("feedback_id", generate_message_id())


            # 👇 Capture corresponding user query
            user_query = ""
            if i > 0 and isinstance(messages[i-1], HumanMessage):
                user_query = messages[i-1].content

            with st.chat_message("assistant", avatar="assets/assistant.png"):

                st.markdown(
                    """
                    <div style="font-size:18px; font-weight:700; margin-bottom:6px;">
                        A360 Response:
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                current_df = None

                # ---------- TEXT ----------
                parsed = parse_agent_output(msg.content)

                if parsed["sql_query"] and st.session_state['role']=="Analyst":
                    st.subheader("SQL Query")
                    st.code(parsed["sql_query"], language="sql")

                if parsed["result_summary"]:
                    st.subheader("Summary")
                    st.write(parsed["result_summary"])

                

                # ---------- COLLECT FOLLOWING RESULT MESSAGES ----------
                j = i + 1

                while j < len(messages) and isinstance(messages[j], AIMessage):

                    sub_msg = messages[j]
                    msg_type = sub_msg.additional_kwargs.get("type")

                    # ----- SQL RESULT -----
                    if msg_type == "sql_result":
                        st.markdown("**Query Results**")
                        current_df = deserialize_df(sub_msg.additional_kwargs["data"])
                        st.dataframe(current_df)

                    # ----- VISUALIZATION -----
                    elif msg_type == "visualization" and sub_msg.additional_kwargs["code"] != "NO_VISUALIZATION":

                        # st.markdown("**Visualization**")

                        if current_df is not None:
                            safe_globals = {"pd": pd, "df": current_df.copy()}
                            safe_locals = {}

                            try:
                                exec(sub_msg.additional_kwargs["code"], safe_globals, safe_locals)
                                fig = safe_locals.get("fig")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Visualization failed: {e}")

                    else:
                        break

                    j += 1

                if parsed["relevant_questions"]:
                    st.subheader("Potential Follow-up Questions")
                    st.write(parsed["relevant_questions"])
                
                # ---------- FEEDBACK (AFTER FINAL OUTPUT) ----------
                already = feedback_exists(st.session_state.thread_id, feedback_id)

                if not already:
                    col1, col2, col3 = st.columns([1, 1, 8])

                    with col1:
                        if st.button("👍", key=f"up_{feedback_id}_{i}"):
                            save_feedback(
                                st.session_state.thread_id,
                                feedback_id,
                                user_query,
                                msg.content,
                                1
                            )
                            st.rerun()

                    with col2:
                        if st.button("👎", key=f"down_{feedback_id}_{i}"):
                            save_feedback(
                                st.session_state.thread_id,
                                feedback_id,
                                user_query,
                                msg.content,
                                -1
                            )
                            st.rerun()
                            
                else:
                    st.markdown("✅ Feedback submitted")


            # Skip processed assistant block
            i = j
            continue

        i += 1


def run_with_status(chatbot, user_msg, config):

    result_container = {}

    def run_agent(local_chatbot, local_user_msg, local_config):
        result_container["response"] = local_chatbot.invoke(
            {"messages": [local_user_msg]},
            config=local_config
        )

    # start thread safely (NO session_state inside)
    t = Thread(target=run_agent, args=(chatbot, user_msg, config))
    t.start()

    steps = [
        "🔍 Analysing your query...",
        "🧠 Understanding intent...",
        "🛠 Generating Code...",
        "📊 Fetching database results...",
        "📈 Creating visualization...",
        "✍️ Preparing summary..."
    ]

    status_box = st.empty()
    i = 0

    dots = ["", ".", "..", "..."]

    while t.is_alive():
        status_box.markdown(
            f"""
            <div style="
                display:inline-block;
                font-size:14px;
                padding:8px 12px;
                border-radius:12px;
                background:#f1f3f5;
                color:#495057;
                margin-top:6px;
                margin-bottom:6px;
                max-width:420px;
                line-height:1.4;
                border:1px solid #e0e0e0;
                font-weight:500;
                ">
                {steps[i % len(steps)]}<span style="opacity:0.6">{dots[i % 4]}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(6.1)
        i += 1

    status_box.empty()
    return result_container["response"]



def render_landing_page():

    
    st.image("assets/a360_3.png", width=200)

    

    st.caption("Your AI Data Analyst — Ask. Analyze. Visualize.")

    st.divider()
    st.subheader("User Docs")
    c1, = st.columns(1)

    with c1:
        st.markdown("[📖 Read FAQs](https://acuitas360llc-my.sharepoint.com/:w:/g/personal/greddy_acuitas360_com/IQDJSsi5qkZIQ5vswPlBLsSHAVC-rm20D1CkqDK3loiPbv0)")
        st.markdown("[📖 Read Example Questions](https://acuitas360llc-my.sharepoint.com/:w:/g/personal/greddy_acuitas360_com/IQDceNCoy-ABRIhboMTBX5w-AXHY336QS7v8_4NfZylpNQU)")
        st.markdown("[📖 Read User Training Documentation](https://acuitas360llc-my.sharepoint.com/:w:/g/personal/greddy_acuitas360_com/IQCe42TnuG3QR522QMCbeo2yARXAk1aZOWej3nF-bIcjP2Q)")
        

    st.divider()
    st.subheader("Try asking:")

    example_questions = {
    "🌍📊 Nation": [
        "Give me sales by week for recent 52 weeks?",
        "How much did sales grow in the recent 4 weeks?",
        "How many uniques weeks of drug sales available in the data",
        "How much drug grown in recent 4 weeks and how that compared with growth in recent 8 weeks"
    ],

    "🗺️📍 Geography": [
        "Provide weekly sales trend by area and nation",
        "Provide weekly sales trend by region and nation",
        "Provide sales contribution by Area and estimate growth in recent 4 weeks compared to previous 4 weeks",
        "Provide sales contribution by region and estimate growth in recent 4 weeks compared to previous 4 weeks"
    ],

    "🏥🏢 Parent Accounts": [
        "What is the sales contribution of parent accounts by BC potential segment",
        "What is the sales contribution of parent accounts which are academic, IDNs and Community",
        "What is the sales growth in recent 4 weeks by parent account type and how does that compare with nation",
        "What is the sales growth in recent 4 weeks by BC potential segment and how does that compare with nation"
    ],

    "🧑‍⚕️🏬 Child Accounts": [
        "What is the sales contribution of child accounts by BC potential segment",
        "What is the sales contribution of child accounts which are academic and Community",
        "What is the sales growth in recent 4 weeks by child account type and how does that compare with nation",
        "What is the sales growth in recent 4 weeks by child BC potential segment and how does that compare with nation"
    ],
}



    question_id = 0

    for category, questions in example_questions.items():

        st.markdown(f"### {category}")   # Category Heading

        for q in questions:
            if st.button(q, key=f"example_{question_id}"):
                st.session_state.prefill_question = q
                st.session_state.app_mode = "chat"
                st.rerun()

            question_id += 1


    st.divider()
    st.info("💡 Tip: Upload a CSV of multiple questions from the sidebar to auto-analyze reports")

# ==========================================================
# ================== FEEDBACK DATABASE ======================
# ==========================================================

def feedback_exists(thread_id, feedback_id):
    conn = st.session_state.checkpointer.conn
    cursor = conn.execute(
        "SELECT 1 FROM message_feedback WHERE thread_id=? AND message_id=? LIMIT 1",
        (thread_id, feedback_id)
    )
    return cursor.fetchone() is not None


def init_feedback_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS message_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT,
        message_id TEXT,
        user_query TEXT,
        assistant_response TEXT,
        rating INTEGER,
        created_at TEXT
    )
    """)
    conn.commit()


def generate_message_id():
    return str(uuid.uuid4())


def save_feedback(thread_id, message_id, user_query, assistant_response, rating):
    conn = st.session_state.checkpointer.conn
    print("Hello from save_feedback 1")
    conn.execute(
        """
        INSERT INTO message_feedback
        (thread_id, message_id, user_query, assistant_response, rating, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            message_id,
            user_query,
            assistant_response,
            rating,
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()
    print("Hello from save_feedback 2")


####################################################################### START CONFIGURATION ################################################################################
if "checkpointer" not in st.session_state:
    conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
    init_feedback_db(conn)  # ### FEEDBACK ###
    st.session_state.checkpointer = SqliteSaver(conn=conn)

if "feedback_given" not in st.session_state:  # ### FEEDBACK ###
    st.session_state.feedback_given = set()

if "chatbot" not in st.session_state:
    st.session_state.chatbot = build_chatbot(
        checkpointer=st.session_state.checkpointer
    )

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=retrieve_all_threads()

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "home"

if "last_ai_message_id" not in st.session_state:
    st.session_state.last_ai_message_id = None

if st.session_state.last_ai_message_id is None:
    st.session_state.last_ai_message_id = generate_message_id()

if "role" not in st.session_state:
    st.session_state["role"] = "Marketing Head"
# =========================================================
# BULK STREAMING STATE
# =========================================================
if "bulk_mode" not in st.session_state:
    st.session_state.bulk_mode = False

if "bulk_questions" not in st.session_state:
    st.session_state.bulk_questions = []

if "bulk_index" not in st.session_state:
    st.session_state.bulk_index = 0

add_thread(st.session_state["thread_id"])




####################################################################### SIDE BAR ################################################################################

# st.sidebar.title("📊 A360")
st.sidebar.image("assets/a360_3.png", width=150)
# st.sidebar.markdown("### A360")
with st.sidebar:
    st.header("Select Role")

    selected_role = st.selectbox(
        "Choose Role",
        ["Analyst", "Marketing Head"],
        index=["Analyst", "Marketing Head"].index(st.session_state["role"])
    )

    st.session_state["role"] = selected_role

if st.sidebar.button("💬 New Chat", type="secondary"):
    reset_chat()
    st.rerun()

#######################################################################
# FAQ
#######################################################################
if st.sidebar.button("📖 Daily Pulse", type="secondary"):
    try:
        st.session_state.app_mode = "chat"
        df_csv = pd.read_csv("FAQ.csv")

        

        # Let user select column
        selected_column = "Questions"

        questions_list = df_csv[selected_column].dropna().tolist()
        chatbot = st.session_state.chatbot
     
            
        config = {"configurable": {"thread_id": st.session_state['thread_id']}}

        st.session_state.app_mode = "chat"
        st.session_state.bulk_mode = True
        st.session_state.bulk_questions = questions_list
        st.session_state.bulk_index = 0

        st.rerun()

       


    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

#######################################################################
# CSV BULK QUESTION UPLOAD
#######################################################################

st.sidebar.subheader("📂 Bulk Question Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with Questions",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        st.session_state.app_mode = "chat"
        df_csv = pd.read_csv(uploaded_file)

        st.sidebar.success("File uploaded successfully")

        # Let user select column
        selected_column = st.sidebar.selectbox(
            "Select Question Column",
            df_csv.columns
        )

        questions_list = df_csv[selected_column].dropna().tolist()

        st.sidebar.write(f"Total Questions Found: {len(questions_list)}")
        if st.sidebar.button("▶ Process All Questions"):

            st.session_state.app_mode = "chat"
            st.session_state.bulk_mode = True
            st.session_state.bulk_questions = questions_list
            st.session_state.bulk_index = 0

            st.rerun()

        
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
#---------------------------------------------Search Conversations-----------------------------------------------------------
search_query = st.sidebar.text_input(
    "🔎 Search Conversations",
    placeholder="Type keyword..."
)

def thread_matches_search(thread_id: str, query: str) -> bool:
    """Return True if any message in thread contains query"""
    if not query:
        return True

    messages = load_conversation(thread_id)
    q = query.lower()

    for msg in messages:
        try:
            if hasattr(msg, "content") and msg.content:
                if q in msg.content.lower():
                    return True
        except:
            pass

    return False

#----------------------------------------------MY Conversations-------------------------------------------------
st.sidebar.header("My Conversations")

sorted_threads = sorted(
    st.session_state["chat_threads"],
    reverse=True
)

# Apply search filter
if search_query:
    sorted_threads = [
        t for t in sorted_threads
        if thread_matches_search(t, search_query)
    ]

if search_query and len(sorted_threads) == 0:
    st.sidebar.info("No conversations found")  
# Create buckets
thread_groups = {
    "Today": [],
    "Yesterday": [],
    "Last 7 Days": [],
    "History": []
}

for thread_id in sorted_threads:
    category = categorize_thread(thread_id)
    thread_groups[category].append(thread_id)

for category, threads in thread_groups.items():

    if not threads:
        continue

    st.sidebar.markdown(f"#### {category}")

    for thread_id in threads:
        messages = load_conversation(thread_id=thread_id)
        title = get_thread_title(messages)
        if search_query and search_query.lower() in title.lower():
            title = f"🔎 {title}"

        if st.sidebar.button(title, key=f"thread_{thread_id}"):
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = load_conversation(thread_id)
            render_history()


st.sidebar.divider()
st.sidebar.subheader("🧪 Feedback DB Test")

if st.sidebar.button("Insert Test Feedback"):
    try:
        save_feedback(
            thread_id="test_thread",
            message_id="test_msg_1",
            user_query="test question",
            assistant_response="test answer",
            rating=1
        )
        st.sidebar.success("Inserted into DB!")
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")

if st.sidebar.button("Show Feedback Table"):
    conn = st.session_state.checkpointer.conn
    df = pd.read_sql_query("SELECT * FROM message_feedback ORDER BY id DESC", conn)
    st.dataframe(df)



#########################################################################################################################################################



 # ---------- HOME SCREEN ----------
if st.session_state.app_mode == "home" and len(st.session_state.message_history) == 0:
    render_landing_page()


# ---------- RENDER CHAT (ALWAYS FROM HISTORY) ----------
if st.session_state.app_mode == "chat":
    render_history()

# =========================================================
# BULK PROCESSOR (One Question Per Rerun)
# =========================================================

if st.session_state.bulk_mode:

    questions = st.session_state.bulk_questions
    index = st.session_state.bulk_index

    if index < len(questions):

        question = questions[index]
        with st.chat_message("user", avatar="assets/success.png"):
            st.markdown(
                '<div style="font-size:18px;font-weight:700;margin-bottom:6px;">User:</div>',
                unsafe_allow_html=True
            )
            st.markdown(question)

        user_msg = HumanMessage(content=question)

        config = {"configurable": {"thread_id": st.session_state['thread_id']}}

       

        # 2️⃣ Run assistant with loading animation
        chatbot = st.session_state.chatbot

        with st.chat_message("assistant", avatar="assets/assistant.png"):
            st.markdown(
                '<div style="font-size:18px;font-weight:700;margin-bottom:6px;">A360 Response:</div>',
                unsafe_allow_html=True
            )
            response = run_with_status(chatbot, user_msg, config)

        # 3️⃣ Attach feedback IDs
        for msg in response["messages"][-4:]:
            if isinstance(msg, AIMessage) and not msg.additional_kwargs:
                msg.additional_kwargs["feedback_id"] = generate_message_id()

        # 4️⃣ Store assistant messages
        st.session_state.message_history.extend(response["messages"][-4:])

        # 5️⃣ Move to next question
        st.session_state.bulk_index += 1

        # 6️⃣ Rerun → This causes incremental display
        st.rerun()

    else:
        # Bulk finished
        st.session_state.bulk_mode = False
        st.session_state.bulk_questions = []
        st.session_state.bulk_index = 0

prefill = st.session_state.pop("prefill_question", "")
user_input = st.chat_input("Ask a question…")

if prefill:
    user_input = prefill


# =========================================================
# HANDLE NEW USER MESSAGE (STATE ONLY — NO UI HERE)
# =========================================================
if user_input:

    st.session_state.app_mode = "chat"

    # 1️⃣ show user message
    with st.chat_message("user", avatar="assets/success.png"):
        st.markdown(
            '<div style="font-size:18px;font-weight:700;margin-bottom:6px;">User:</div>',
            unsafe_allow_html=True
        )
        st.markdown(user_input)

    # 2️⃣ create message
    user_msg = HumanMessage(content=user_input)
    config = {"configurable": {"thread_id": st.session_state['thread_id']}}

    # 3️⃣ assistant placeholder + live status
    with st.chat_message("assistant", avatar="assets/assistant.png"):
        st.markdown(
            '<div style="font-size:18px;font-weight:700;margin-bottom:6px;">A360:</div>',
            unsafe_allow_html=True
        )
        chatbot = st.session_state.chatbot   # <-- extract BEFORE thread
        response = run_with_status(chatbot, user_msg, config)

    # 4️⃣ attach feedback ids
    for msg in response["messages"][-4:]:
        if isinstance(msg, AIMessage) and not msg.additional_kwargs:
            msg.additional_kwargs["feedback_id"] = generate_message_id()

    # 5️⃣ store messages
    st.session_state.message_history.extend(response["messages"][-4:])

    # 6️⃣ redraw final UI
    st.rerun()

        

    
    # print("Message History")
    # print("-"*100)
    # print(st.session_state.message_history)
    # print()
    
