# 1. Import AI modules BEFORE Streamlit to bypass custom web import hooks
import ai_router

import streamlit as st
import db_manager
import uuid

# 2. Cache the pre-loaded router so it doesn't reboot on every chat message
@st.cache_resource
def get_router():
    return ai_router

ai = get_router()

db_manager.init_db()

if "session_id" not in st.session_state:
    st.session_state.session_id = f"web_session_{str(uuid.uuid4())[:6]}"

st.set_page_config(page_title="Local AI Router", page_icon="⚡")
st.title("Edge-Computing LoRA Hot-Swapper")

st.sidebar.title("Routing Control")
st.sidebar.markdown("Select the active LoRA adapter. The 130M Mamba model will dynamically swap matrix weights on the fly.")
active_persona = st.sidebar.radio("Active Persona", ["code", "clinical", "legal"])

history = db_manager.get_chat_history(st.session_state.session_id)
for msg in history:
    with st.chat_message(msg["sender"].lower()):
        if msg["sender"] == "Assistant":
            st.markdown(f"**[{msg['persona'].upper()}]** {msg['content']}")
            st.caption(f"⏱️ Latency: {msg['latency_seconds']}s")
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Test a prompt..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    db_manager.log_message(st.session_state.session_id, "User", "None", prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Routing to {active_persona.upper()} brain..."):
            reply, latency = ai.generate_response(active_persona, prompt)
            
            st.markdown(f"**[{active_persona.upper()}]** {reply}")
            st.caption(f"⏱️ Hot-Swap Latency: {latency}s")
            
    db_manager.log_message(st.session_state.session_id, "Assistant", active_persona, reply, latency)