import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from load_database import retrieve_past_conversations, get_last_user_message
from sentence_transformers import SentenceTransformer, util

def load_pretrained_model():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("alvinwongster/LuminAI")
    model = AutoModelForCausalLM.from_pretrained("alvinwongster/LuminAI")

    model.to(device)

    return model, tokenizer, device 

def get_sentiment(text):
    crisis_keywords = ["end my life", "suicide", "don't want to live", "kill myself", "worthless", "no reason to live", "want to die"]
    if any(phrase in text.lower() for phrase in crisis_keywords):
        return "crisis"  

    sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_classifier(text)[0]
    label = result['label']

    if label == "NEGATIVE":
        return "negative"
    elif label == "POSITIVE":
        return "positive"
    else:
        return "neutral"

def translate_cn_to_en(text):
    pipe = pipeline("text2text-generation", model="Varine/opus-mt-zh-en-model")
    translated_output = pipe(text)[0]  # Extracting the first result
    translated_text = translated_output.get("generated_text", "Translation failed")  # Use .get() to avoid KeyError
    return translated_text

def translate_en_to_cn(reply): 
    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
    translated_text = pipe(reply)[0]['translation_text']

    return translated_text

def chatbot_response(prompt, connection, cursor):
    sentiment_results = get_sentiment(prompt)

    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    predefined_responses = {
        "What can you do?": "I am Lumin.AI, your therapist chatbot. I am here to provide 24/7 support by listening and offering resources for your mental well-being. Feel free to talk to me about anything that is affecting your mental health!",
        "How can you help?": "I am Lumin.AI, your therapist chatbot. I am here to provide 24/7 support by listening and offering resources for your mental well-being. Feel free to talk to me about anything that is affecting your mental health!",
        "What are you?": "I am Lumin.AI, your therapist chatbot. I am here to provide 24/7 support by listening and offering resources for your mental well-being. Feel free to talk to me about anything that is affecting your mental health!",
        "What is your role?": "I am Lumin.AI, your therapist chatbot. I am here to provide 24/7 support by listening and offering resources for your mental well-being. Feel free to talk to me about anything that is affecting your mental health!",
        "What are the help lines for mental health issues?": """Here are some helplines:\n- SOS Hotline: Call 1767\n- Mindline: https://www.mindline.sg/\n- MindSG: https://www.healthhub.sg/programmes/mindsg/discover\n- SAMH: https://www.samhealth.org.sg/""",
        "What resources are available?": """Here are mental health resources:\n- SOS Hotline: Call 1767\n- Mindline: https://www.mindline.sg/\n- MindSG: https://www.healthhub.sg/programmes/mindsg/discover\n- SAMH: https://www.samhealth.org.sg/""",
        "Who can I talk to?": """Here are some mental health helplines:\n- SOS Hotline: Call 1767\n- Mindline: https://www.mindline.sg/\n- MindSG: https://www.healthhub.sg/programmes/mindsg/discover\n- SAMH: https://www.samhealth.org.sg/""",
        "Where can I find help?": """Here are support hotlines:\n- SOS Hotline: Call 1767\n- Mindline: https://www.mindline.sg/\n- MindSG: https://www.healthhub.sg/programmes/mindsg/discover\n- SAMH: https://www.samhealth.org.sg/"""
    }

    model, tokenizer, device = load_pretrained_model()
    retrieved_context = retrieve_past_conversations(prompt, connection, cursor)

    system_prompt = "You are a helpful and supportive chatbot. Answer the user's question with empathy and clarity, without repeating their words exactly."
    
    # Compute similarity with predefined responses
    user_embedding = similarity_model.encode(prompt, convert_to_tensor=True)
    
    best_match = None
    best_score = 0

    for key in predefined_responses:
        key_embedding = similarity_model.encode(key, convert_to_tensor=True)
        score = util.pytorch_cos_sim(user_embedding, key_embedding).item()  # Cosine similarity score
        
        if score > best_score:
            best_score = score
            best_match = key

    # If similarity score is above 0.7 (70%), return predefined response
    if best_score >= 0.7:
        return predefined_responses[best_match], "neutral"
    
    last_bot_question = get_last_user_message(connection, cursor)
    
    if last_bot_question:
        confirmation_phrases = ["yes", "sure", "I need that", "okay", "please", "go ahead"]
        negative_phrases = ["no", "not now", "maybe later"]

        for phrase in confirmation_phrases:
            if phrase in prompt.lower():
                retrieved_context = last_bot_question
                sentiment_results = "neutral"

        for phrase in negative_phrases:
            if phrase in prompt.lower():
                return "Alright, let me know if you need help with anything else!", "neutral"
    try:
        full_prompt = f"{system_prompt}\n{retrieved_context}\nUser: {prompt}\nBot:"
    except Exception as e: 
        print(f"full prompt error: {e}")

    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    outputs = model.generate(
        **inputs, 
        max_length=650,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,  
        temperature=0.8,  
        top_p=0.9,  
        top_k=50  
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Ensure the response does not include the system prompt
    response = response.replace(system_prompt, "").strip()
    
    # Remove any leftover system prompt instructions
    if "Bot:" in response:
        response = response.split("Bot:")[-1].strip()

    return response, sentiment_results