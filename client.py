import os
import sys

# Safeguard against protobuf _upb metaclass incompatibility on Python >= 3.14
if sys.version_info >= (3, 14):
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from rag import RAG
from dotenv import load_dotenv
import time


# 1. Setup Page Configuration
st.set_page_config(page_title="RAG-CRAW", page_icon="🕸️", layout="centered", initial_sidebar_state="expanded")

# 2. Load API Key
load_dotenv('.env')
DEFAULT_API_KEY = os.getenv('GOOGLE_API_KEY', '')
if not DEFAULT_API_KEY:
    try:
        DEFAULT_API_KEY = st.secrets.get('GOOGLE_API_KEY', '')
    except Exception:
        pass


# 3. Custom CSS for UI Polish
ui_styling = """
<style>
/* 1. Safely hide the top-right menu and deploy buttons */
#MainMenu {visibility: hidden;}
.stAppDeployButton {display: none;}
[data-testid="stToolbar"] {display: none;} 
[data-testid="stHeader"] {background-color: transparent !important;}

/* 2. Lock the sidebar open ON DESKTOP ONLY. */
@media (min-width: 768px) {
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
}

/* 3. Keep original sidebar footer & button styling */
.sidebar-footer {
    margin-top: 40px; 
    padding-top: 15px;
    text-align: center;
    font-size: 13px;
    color: #888;
    border-top: 1px solid #444;
}
.sidebar-footer a { color: #6366f1; text-decoration: none; font-weight: 600; }
.sidebar-footer a:hover { text-decoration: underline; }
.stButton>button[kind="primary"] {
    background-color: #6366f1; color: white; border-radius: 8px; border: none; transition: all 0.2s ease-in-out;
}
.stButton>button[kind="primary"]:hover { background-color: #4f46e5; transform: translateY(-1px); }
.streamlit-expanderHeader { font-size: 14px; font-weight: 500; color: #a1a1aa; }
</style>
"""
st.markdown(ui_styling, unsafe_allow_html=True)

# 4. Initialize Session States
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'current_url' not in st.session_state:
    st.session_state['current_url'] = ""
if 'resource_processor' not in st.session_state:
    st.session_state['resource_processor'] = None
if 'crawl_logs' not in st.session_state:
    st.session_state['crawl_logs'] = [] 
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "👋 Hello! Please load a website URL from the sidebar to begin.", "sources": []}]

# --- SIDEBAR: CONFIGURATION & LOGS ---
with st.sidebar:
    st.title("⚙️ Setup")
    
    api_key_input = st.text_input(
        "Google Gemini API Key:",
        value=st.session_state.get("user_api_key", DEFAULT_API_KEY),
        type="password",
        placeholder="Paste your AI Studio key...",
        help="Obtain key from https://aistudio.google.com/app/apikey"
    )
    st.session_state["user_api_key"] = api_key_input.strip()

    website_url = st.text_input("Enter website URL:", placeholder="https://en.wikipedia.org/wiki/Virat_Kohli")
    
    if st.button("Load & Vectorize Website", type="primary", use_container_width=True):
        active_key = st.session_state.get("user_api_key") or DEFAULT_API_KEY
        target_url = website_url.strip()
        
        if not target_url:
            st.warning("Please enter a website URL first.")
        elif not active_key:
            st.error("Google Gemini API Key is missing! Enter it above or set GOOGLE_API_KEY in .env.")
        else:
            if not target_url.startswith(('http://', 'https://')):
                target_url = 'https://' + target_url

            st.session_state['crawl_logs'] = []
            st.session_state['messages'] = [{"role": "assistant", "content": f"✅ Successfully connected to `{target_url}`. Ask me anything!", "sources": []}]
            
            with st.status("Initializing RAG Engine...", expanded=True) as status:
                def log_update(msg):
                    status.write(msg)
                    st.session_state['crawl_logs'].append(msg)
                
                try:
                    st.session_state['resource_processor'] = RAG(target_url, active_key, write_function=log_update)
                    st.session_state['data_loaded'] = True
                    st.session_state['current_url'] = target_url
                    status.update(label="System Ready!", state="complete", expanded=False)
                    st.toast("Database loaded!", icon="🚀")
                    time.sleep(1.0)
                    st.rerun() 
                except Exception as e:
                    status.update(label="Failed to process website.", state="error")
                    st.error(f"Error details: {e}")

    if st.session_state['data_loaded'] and st.session_state['crawl_logs']:
        with st.expander("Terminal Logs", expanded=False):
            for log in st.session_state['crawl_logs']:
                st.markdown(f"<span style='font-size: 12px; color: #a1a1aa;'>{log}</span>", unsafe_allow_html=True)
                
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True) and st.session_state['data_loaded']:
        st.session_state['messages'] = [{"role": "assistant", "content": f"Chat cleared! Ask me anything else about `{st.session_state['current_url']}`.", "sources": []}]
        st.rerun()

    st.markdown("""
    <div class="sidebar-footer">
        <div style="margin-bottom: 5px;">Built by <b>Krish Agarwal</b></div>
        <div style="font-size: 11px; margin-bottom: 10px;">Jadavpur University IT '28</div>
        <a href="https://github.com/krishjuit" target="_blank">GitHub</a> • 
        <a href="www.linkedin.com/in/krish-agarwal-b67b57321" target="_blank">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)


# --- MAIN CHAT AREA ---
st.title("🕸️ RAG-CRAW")

if st.session_state['data_loaded']:
    st.caption(f"🟢 **Active Session:** Chatting with `{st.session_state['current_url']}`")
else:
    st.caption("🔴 **Status:** Awaiting URL Input (See Sidebar)")

st.divider()

# Render chat messages with their respective Sources
for msg in st.session_state['messages']:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # If it's an assistant message and has sources, render the expander
        if msg.get("sources"):
            with st.expander("📚 View Sources Used"):
                for i, source in enumerate(msg["sources"]):
                    st.markdown(f"**Source {i+1}**")
                    st.info(source)

prompt_placeholder = "Ask a question about the loaded website..." if st.session_state['data_loaded'] else "👈 Load a website in the sidebar first!"

if prompt := st.chat_input(prompt_placeholder, disabled=not st.session_state['data_loaded']):
    
    st.session_state['messages'].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            
            # The backend now returns a dictionary with 'answer' and 'sources'
            result_dict = st.session_state['resource_processor'].get_response(prompt)
            answer_text = result_dict["answer"]
            sources_list = result_dict["sources"]
            
            # Display the answer
            st.write(answer_text)
            
            # Render the sources immediately for the live message
            if sources_list:
                with st.expander("📚 View Sources Used"):
                    for i, source in enumerate(sources_list):
                        st.markdown(f"**Source {i+1}**")
                        st.info(source)
            
            # Save both to chat history
            st.session_state['messages'].append({
                "role": "assistant", 
                "content": answer_text,
                "sources": sources_list
            })