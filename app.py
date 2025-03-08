import streamlit as st
from chatbot import chatbot_response
from load_database import start_database, store_conversation, clear_conversations
from notify import trigger_alert

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "connection" not in st.session_state:
    st.session_state.connection = None
if "cursor" not in st.session_state:
    st.session_state.cursor = None


print("welcome")

# Streamlit page setup
st.title("🤖 Chatbot")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

if st.session_state.cursor is None and st.session_state.connection is None:
    st.session_state.connection, st.session_state.cursor = start_database()
    print("Database loaded")
    clear_conversations(st.session_state.connection, st.session_state.cursor)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]): 
        st.markdown(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    bot_reply, sentiments = chatbot_response(user_input, st.session_state.connection, st.session_state.cursor)
    store_conversation(user_input, bot_reply, sentiments, st.session_state.connection, st.session_state.cursor)

    if sentiments == "crisis":
        recipient = "alvinwongster@gmail.com" # therapist email
        trigger_alert(user_input, recipient)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(bot_reply)

    # Rerun to maintain chat history
    st.rerun()