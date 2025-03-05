from sentence_transformers import SentenceTransformer
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os

def start_database():
    load_dotenv()
    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

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
    except Exception as e:
        print(e)


def store_conversation(user_message, bot_response, sentiment, connection, cursor):
    """Stores a conversation in Supabase PostgreSQL."""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = embedding_model.encode([user_message])[0].tolist()
    try:
        cursor.execute(
            "INSERT INTO conversations (user_message, bot_response, sentiment, embedding) VALUES (%s, %s, %s, %s)",
            (user_message, bot_response, sentiment, embedding)
        )
        connection.commit()
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