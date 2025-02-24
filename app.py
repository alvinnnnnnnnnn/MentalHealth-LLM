import shelve
import streamlit as st
import torch
import chromadb

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


chroma_client = chromadb.PersistentClient(path="./chroma_db")
chat_collection = chroma_client.get_or_create_collection(name="chat_history")

st.title("Mental Health Chatbot")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("results", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("results", device_map="auto")

# Store in session state
if "model" not in st.session_state:
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer

# Function to load chat history
def load_chat_history(): 
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Function to save chat history
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & efficient

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def store_embedded_chat(message):
    embedding = get_embedding(message)
    
    chat_collection.add(
        embeddings=[embedding],  # Vector representation
        metadatas=[{"message": message}]
    )

def retrieve_similar_chats(query, top_k=5):
    query_embedding = get_embedding(query)

    results = chat_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    return [(res["message"], res["response"]) for res in results["metadatas"][0]]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar for deleting chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat history
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    past_chats = retrieve_similar_chats(prompt, top_k=5)
    context = "\n".join([f"User (Past): {chat[0]}\nBot: {chat[1]}" for chat in past_chats])
    full_prompt = f"{context}\nUser (Now): {prompt}\nBot:"

    # Assistant response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""

        # Tokenize input
        input_ids = st.session_state.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        # Generate response with streaming effect
        with torch.no_grad():
            output = st.session_state.model.generate(input_ids, max_length=150, do_sample=True)
        
        # Decode and display response
        generated_text = st.session_state.tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        for char in generated_text:
            full_response += char
            message_placeholder.markdown(full_response + "|")  # Simulate typing effect
        
        message_placeholder.markdown(full_response)

    # Save message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat_history(st.session_state.messages)
    store_embedded_chat(prompt, full_response)
