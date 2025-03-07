import smtplib
import streamlit as st
from email.message import EmailMessage

def trigger_alert(user_input, to):
    user = 'mentalhealth.llm@gmail.com'
    password = st.secrets['APP_PASSWORD']
    body = f"Patient has mentioned '{user_input}'"
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = "Crisis Detected"
    msg['From'] = user
    msg['To'] = to

    server = smtplib.SMTP_SSL('smtp.gmail.com', 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)

    server.quit()