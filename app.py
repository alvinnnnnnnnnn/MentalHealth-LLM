import shelve
import streamlit as st
import torch
import chromadb
import time

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chat_collection = chroma_client.get_or_create_collection(name="chat_history")

st.title("Mental Health Chatbot")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        body, .stApp {
            font-family: 'Poppins', sans-serif;
            color: #333;
        }

        .stSidebar {
            font-family: 'Poppins', sans-serif;
            background-color: #d1f0e6;
            border-radius: 15px;
            padding: 10px;
        }

        .stButto>button {
            background-color: #333
            color: #FFFFFF;
        }  

        .stButton>button:hover, .stButton>button:focus, .stButton>button:active {
            background-color: #333 !important; /* Change this to your desired hover color */
            color: #FFFFFF;
            border: none;
        }

        .stExpander {
            background-color: #FFFFFF;
            border-radius: 10px;
        }        
    </style>
""", unsafe_allow_html=True)


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
    with shelve.open("chat_history", writeback=True) as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history", writeback=True) as db:
        db["messages"] = messages


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & efficient

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def store_embedded_chat(user_message, bot_response):
    embedding = get_embedding(user_message)
    
    # Generate a unique ID (simple approach using timestamp)
    import uuid
    unique_id = str(uuid.uuid4())
    
    chat_collection.add(
        ids=[unique_id],  # Required parameter
        embeddings=[embedding],
        metadatas=[{"message": user_message, "response": bot_response}]
    )

def retrieve_similar_chats(query, top_k=5):
    query_embedding = get_embedding(query)

    results = chat_collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k
    )

    # Check if any results exist before accessing
    if results.get("metadatas") and len(results["metadatas"]) > 0:
        return [(item.get("message", ""), item.get("response", "")) 
                for item in results["metadatas"][0] if item]
    return []


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

if "delete_confirmed" not in st.session_state:
    st.session_state.delete_confirmed = False

# Sidebar for deleting chat history
with st.sidebar:

    st.header("Share your thoughts with us!")

    with st.expander("Mental Health Resources"):
        st.write('''
            -  **SOS Hotline**: Call 1767
            -  [**Mindline Website**](https://www.mindline.sg/)
            -  [**MindSG Website**](https://www.healthhub.sg/programmes/mindsg/discover)
            -  [**SAMH Website**](https://www.samhealth.org.sg/)
         ''')


    # Delete Chat History
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])


# Display chat history
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment(text):
    """Improved sentiment analysis with keyword-based distress detection."""
    crisis_keywords = ["end my life", "suicide", "don't want to live", "kill myself", "worthless", "no reason to live"]

    # Check if crisis words are in the input
    if any(phrase in text.lower() for phrase in crisis_keywords):
        return "crisis"  # Override sentiment if crisis words are detected

    # Otherwise, use DistilBERT-based sentiment analysis
    result = sentiment_classifier(text)[0]
    label = result['label']

    # Convert to sentiment categories based on DistilBERT outputs
    if label == "NEGATIVE":
        return "negative"
    elif label == "POSITIVE":
        return "positive"
    else:
        return "neutral"

device = st.session_state.model.device  # Get the model's actual device

def chatbot_response(prompt, max_length=200):
    system_prompt = "You are a helpful and supportive chatbot. Answer the user's question in a clear and concise way without repeating their words exactly."
    full_prompt = f"{system_prompt}\nUser: {prompt}\nBot:"

    sentiment_results = get_sentiment(prompt)

    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    outputs = model.generate(
        **inputs, 
        max_length=max_length,
        repetition_penalty=1.3,  # Penalize repeated words/phrases
        no_repeat_ngram_size=3,  # Prevent 3-word repetition
        temperature=0.7,  # Lower randomness to keep responses meaningful
        top_p=0.9,  # Diverse sampling to avoid looping responses
        top_k=50  # Limits token choices to avoid repetition
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Ensure the response does not include the system prompt
    response = response.replace(system_prompt, "").strip()
    
    # Remove any leftover system prompt instructions
    if "Bot:" in response:
        response = response.split("Bot:")[-1].strip()

    # Translate response to Chinese
    translated = pipe(response)[0]['translation_text']

    return response, translated, sentiment_results

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

        # # Tokenize input
        # device = st.session_state.model.device  # Get the model's actual device
        # input_ids = st.session_state.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Generate response with streaming effect
        # with torch.no_grad():
        #     output = st.session_state.model.generate(input_ids, max_length=150, do_sample=True)
        
        # Decode and display response
        # generated_text = st.session_state.tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        response, translated, sentiment_results = chatbot_response(prompt)

        # for char in generated_text:
        #     full_response += char
        #     message_placeholder.markdown(full_response)
        #     time.sleep(0.02)  # Small delay for smoother effect
        
        for char in response:
            full_response += char
            message_placeholder.markdown(full_response)
            time.sleep(0.02)  # Small delay for smoother effect
        
        # message_placeholder.markdown(full_response)
        message_placeholder.markdown(response)

    # Save message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat_history(st.session_state.messages)
    store_embedded_chat(prompt, full_response)
