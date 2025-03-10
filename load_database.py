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
    limit = 3
    cursor.execute(
        "SELECT timestamp, user_message, bot_response FROM conversations "
        "ORDER BY embedding <-> %s::vector LIMIT %s",
        (query_embedding, limit)
    )

    results = cursor.fetchall()

    if not results:
        return ""

    context = "\n".join([f"[{r[0]}] User: {r[1]}\nBot: {r[2]}" for r in results])

    # max_tokens=300
    # tokenized_context = tokenizer.encode(context, add_special_tokens=False)
    # if len(tokenized_context) > max_tokens:
    #     tokenized_context = tokenized_context[-max_tokens:]  # Keep most recent
    #     context = tokenizer.decode(tokenized_context)

    return context

def get_last_user_message(connection, cursor):
    cursor.execute(
        "SELECT user_message FROM conversations ORDER BY timestamp DESC LIMIT 1;"
    )
    result = cursor.fetchone()
    return result[0] if result else None  # Return the last message if found
