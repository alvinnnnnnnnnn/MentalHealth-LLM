import streamlit as st
from chatbot import chatbot_response
from load_database import start_database, store_conversation, clear_conversations

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit page setup
st.title("🤖 Chatbot")

i=0
if i==0:
    connection, cursor = start_database()
    clear_conversations(connection, cursor)
    i+=1

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]): 
        st.markdown(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_reply, sentiments = chatbot_response(user_input, connection, cursor)
    store_conversation(user_input, bot_reply, sentiments, connection, cursor)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    # Rerun to maintain chat history
    st.rerun()

