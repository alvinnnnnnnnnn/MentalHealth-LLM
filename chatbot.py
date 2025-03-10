import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from load_database import retrieve_past_conversations
from sentence_transformers import SentenceTransformer, util

def load_pretrained_model():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained("results", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("results", device_map="auto")

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

# def chatbot_response(prompt, connection, cursor):
#     model, tokenizer, device = load_pretrained_model()
#     retrieved_context = retrieve_past_conversations(prompt, connection, cursor)

#     system_prompt = "You are a helpful and supportive chatbot. Answer the user's question in a clear and concise way without repeating their words exactly."
#     full_prompt = f"{system_prompt}\n{retrieved_context}\nUser: {prompt}\nBot:"

#     sentiment_results = get_sentiment(prompt)

#     inputs = tokenizer(full_prompt, return_tensors="pt")
#     inputs = {key: val.to(device) for key, val in inputs.items()}

#     outputs = model.generate(
#         **inputs, 
#         max_new_tokens=650,
#         repetition_penalty=1.3,
#         no_repeat_ngram_size=3,  
#         temperature=0.3,  
#         top_p=0.9,  #
#         top_k=50  
#     )

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Ensure the response does not include the system prompt
#     response = response.replace(system_prompt, "").strip()
    
#     # Remove any leftover system prompt instructions
#     if "Bot:" in response:
#         response = response.split("Bot:")[-1].strip()

#     return response, sentiment_results

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
    # Load the sentence transformer model for semantic similarity
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Predefined responses
    predefined_responses = {
        "What can you do?": "I am Lumin.AI, your therapist chatbot aimed to provide 24/7 support for you.",
        "How can you help?": "I am Lumin.AI, your therapist chatbot aimed to provide 24/7 support for you.",
        "What do you do?": "I am Lumin.AI, your therapist chatbot aimed to provide 24/7 support for you.",
        "What are the help lines for depression?": """Here are some helplines and resources you can reach out to:
        
    - **SOS Hotline**: Call 1767
    - [**Mindline Website**](https://www.mindline.sg/)
    - [**MindSG Website**](https://www.healthhub.sg/programmes/mindsg/discover)
    - [**SAMH Website**](https://www.samhealth.org.sg/)""",
        "What resources are available?": """Here are some helplines and resources you can reach out to:
        
    - **SOS Hotline**: Call 1767
    - [**Mindline Website**](https://www.mindline.sg/)
    - [**MindSG Website**](https://www.healthhub.sg/programmes/mindsg/discover)
    - [**SAMH Website**](https://www.samhealth.org.sg/)""",
        "Who can I talk to?": """Here are some helplines and resources you can reach out to:
        
    - **SOS Hotline**: Call 1767
    - [**Mindline Website**](https://www.mindline.sg/)
    - [**MindSG Website**](https://www.healthhub.sg/programmes/mindsg/discover)
    - [**SAMH Website**](https://www.samhealth.org.sg/)"""
    }

    model, tokenizer, device = load_pretrained_model()
    retrieved_context = retrieve_past_conversations(prompt, connection, cursor)

    system_prompt = "You are a helpful and supportive chatbot. Answer the user's question with empathy, and in a clear and concise way without repeating their words exactly."
    
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

    # Otherwise, continue with the chatbot model response
    full_prompt = f"{system_prompt}\n{retrieved_context}\nUser: {prompt}\nBot:"

    sentiment_results = get_sentiment(prompt)

    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    outputs = model.generate(
        **inputs, 
        max_length=350,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,  
        temperature=0.3,  
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