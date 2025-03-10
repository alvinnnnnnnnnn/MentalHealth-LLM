# import streamlit as st
# from chatbot import chatbot_response, translate_cn_to_en, translate_en_to_cn
# from load_database import start_database, store_conversation, clear_conversations
# from notify import trigger_alert

# # Initialize chat history in session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "connection" not in st.session_state:
#     st.session_state.connection = None
# if "cursor" not in st.session_state:
#     st.session_state.cursor = None
# if "language" not in st.session_state:
#     st.session_state.language = "English"


# print("welcome")

# # Streamlit page setup
# st.title("🤖 Lumin.AI")

# if st.session_state.cursor is None and st.session_state.connection is None:
#     st.session_state.connection, st.session_state.cursor = start_database()
#     print("Database loaded")
#     clear_conversations(st.session_state.connection, st.session_state.cursor)

# language = st.selectbox("Select Language", ["English", "Chinese"], index=0)
# st.session_state.language = language

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]): 
#         st.markdown(message["content"])

# user_input = st.chat_input("Type your message...")

# if user_input:
#     if st.session_state.language == "Chinese":
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)

#         user_input_translated = translate_cn_to_en(user_input)
        

#         bot_reply, sentiments = chatbot_response(user_input_translated, st.session_state.connection, st.session_state.cursor)
#         store_conversation(user_input_translated, bot_reply, sentiments, st.session_state.connection, st.session_state.cursor)

#         if sentiments == "crisis":
#             recipient = "alvinwongster@gmail.com" # therapist email
#             trigger_alert(user_input_translated, recipient)

#         bot_reply_translated = translate_en_to_cn(bot_reply)
#         st.session_state.messages.append({"role": "assistant", "content": bot_reply_translated})

#         with st.chat_message("assistant"):
#             st.markdown(bot_reply)

#     else:
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)
#         bot_reply, sentiments = chatbot_response(user_input, st.session_state.connection, st.session_state.cursor)
#         store_conversation(user_input, bot_reply, sentiments, st.session_state.connection, st.session_state.cursor)

#         if sentiments == "crisis":
#             recipient = "alvinwongster@gmail.com" # therapist email
#             trigger_alert(user_input, recipient)

#         st.session_state.messages.append({"role": "assistant", "content": bot_reply})

#         with st.chat_message("assistant"):
#             st.markdown(bot_reply)

#     # Rerun to maintain chat history
#     st.rerun()

import streamlit as st
from chatbot import chatbot_response, translate_cn_to_en, translate_en_to_cn
from load_database import start_database, store_conversation, clear_conversations
from notify import trigger_alert

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "connection" not in st.session_state:
    st.session_state.connection = None
if "cursor" not in st.session_state:
    st.session_state.cursor = None
if "language" not in st.session_state:
    st.session_state.language = "English"

# Streamlit page setup
st.title("🤖 Lumin.AI")

if st.session_state.cursor is None and st.session_state.connection is None:
    st.session_state.connection, st.session_state.cursor = start_database()
    clear_conversations(st.session_state.connection, st.session_state.cursor)

language = st.selectbox("Select Language", ["English", "Chinese"], index=0)
st.session_state.language = language

# Display chat history or default message
if not st.session_state.messages:
    st.markdown("**What can I help with?**")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): 
            st.markdown(message["content"])

user_input = st.chat_input("Type your message...")
# user_input = st.text_area("Type your message...", height=68)

# Display warning below the chat box
st.markdown("---")
st.markdown("⚠️ *Lumin.AI can make mistakes. Please clarify with your therapist if you are unsure.*")

if user_input:
    if st.session_state.language == "Chinese":
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        user_input_translated = translate_cn_to_en(user_input)
        bot_reply, sentiments = chatbot_response(user_input_translated, st.session_state.connection, st.session_state.cursor)
        store_conversation(user_input_translated, bot_reply, sentiments, st.session_state.connection, st.session_state.cursor)

        if sentiments == "crisis":
            recipient = "alvinwongster@gmail.com" # therapist email
            trigger_alert(user_input_translated, recipient)

        bot_reply_translated = translate_en_to_cn(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply_translated})

        with st.chat_message("assistant"):
            st.markdown(bot_reply_translated)

    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        bot_reply, sentiments = chatbot_response(user_input, st.session_state.connection, st.session_state.cursor)
        store_conversation(user_input, bot_reply, sentiments, st.session_state.connection, st.session_state.cursor)

        if sentiments == "crisis":
            recipient = "alvinwongster@gmail.com" # therapist email
            trigger_alert(user_input, recipient)

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        with st.chat_message("assistant"):
            st.markdown(bot_reply)

    # Rerun to maintain chat history
    st.rerun()
