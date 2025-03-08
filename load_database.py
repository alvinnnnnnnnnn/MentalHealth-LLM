from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st
import psycopg2
import os

def start_database():
    USER = st.secrets["user"]
    PASSWORD = st.secrets["password"]
    HOST = st.secrets["host"]
    PORT = st.secrets["port"]
    DBNAME = st.secrets["dbname"]

    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )        
        # Create a cursor to execute SQL queries
        cursor = connection.cursor()

    except Exception as e:
        print(f"Failed to connect: {e}")
    
    return connection, cursor

def clear_conversations(connection, cursor):
    try:
        cursor.execute("DELETE FROM conversations")
        connection.commit()
        print("Database Cleared")
    except Exception as e:
        print(e)


def store_conversation(user_message, bot_response, sentiment, connection, cursor):
    print("storing conversation")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = embedding_model.encode([user_message])[0].tolist()
    try:
        cursor.execute(
            "INSERT INTO conversations (user_message, bot_response, sentiment, embedding) VALUES (%s, %s, %s, %s)",
            (user_message, bot_response, sentiment, embedding)
        )
        connection.commit()
        print("conversation stored")
    except Exception as e:
        print(e)

def retrieve_past_conversations(query, connection, cursor):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode([query])[0] 

    if isinstance(query_embedding, np.ndarray): 
        query_embedding = query_embedding.tolist()

    cursor.execute(
        "SELECT timestamp, user_message, bot_response FROM conversations "
        "ORDER BY embedding <-> %s::vector LIMIT 5",
        (query_embedding,)  # Ensure it's passed as a tuple
    )

    results = cursor.fetchall()

    if results:
        context = "\n".join([f"[{r[0]}] User: {r[1]}\nBot: {r[2]}" for r in results])
        return context
    else:
        return ""