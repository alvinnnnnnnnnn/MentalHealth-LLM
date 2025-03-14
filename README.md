# Lumin.AI 🧠👨‍⚕️

Welcome aboard! **Lumin.AI** is a supportive AI assistant designed to provide immediate emotional support to individuals outside of regular consulting hours. It acts as a supplementary tool for patients and therapists, ensuring that mental health care is more accessible and responsive to users' needs.

The chatbot is designed to offer **24/7 availability, guided therapy techniques, and mood tracking**, helping users manage their mental well-being effectively while maintaining a connection to professional care.

**Disclaimer**: Lumin.AI has not yet been approved by a medical professional, and can make mistakes. Please consult a therapist if you are unsure.

## Features
- 24/7 Availability – The chatbot is always online to assist users.
- Guided Therapy Techniques – Offers cognitive behavioral therapy (CBT)-inspired interactions.
- Mood Tracking – Monitors user sentiment over time and suggests helpful interventions.
- Semantic Memory – Remembers past interactions for a more personalized experience.
- Harm Detection & Therapist Alerts – Detects harmful messages and notifies therapists when needed.

---

## Table of Contents
- [Quick Start](#quick-start)
- [Datasets Used](#datasets-used)
- [Solution](#solution)
- [Model Metrics](#model-metrics)

---

## Quick Start
If you just want to use the chatbot, you can head down to this [link](https://luminai-chatbot.streamlit.app/)

If you want to find out more about the solution, you can follow these simple steps (Ensure that you have a GPU enabled device):
1. Install anaconda from [here](https://www.anaconda.com/) 
2. Install CUDA Toolkit from [here](https://developer.nvidia.com/cuda-toolkit) 
3. Install Git
4. Run the following lines in your terminal (either IDE or device)
```bash
# Clone the repository
git clone https://github.com/alvinnnnnnnnnn/MentalHealth-LLM.git

# Create a conda environment
conda create -n ENV_NAME python=3.9

# Activate the environment
conda activate ENV_NAME`

# Installing / Upgrading required libraries 
pip install --upgrade -r requirements.txt
```
5. Install [pytorch](https://pytorch.org/) based on your OS

## Datasets Used
The chatbot has been trained using conversational data, which is supposed to mimick the patient and the therapist. 5 topics where chosen, and 100 conversations from each of these topics were gathered:
- General 
- Relationships 
- Insecurities
- Victim Mentality 
- Self-Improvement 


## Solution
Used `SmolLM2-360M-Instruct` from Hugging Face, and trained it on the custom dataset from above. 

Visit this [page](https://huggingface.co/alvinwongster/LuminAI) for the uploaded model on Hugging Face, because the final model results are too large to post onto github

## Model Metrics
To evaluate the chatbot's performance based on our use case, the following weighted metrics system was used:
- Empathy Score (40%): 
    - Measures how well the chatbot responds with empathy.
- Human-Likeness Score (20%):
    - Assesses how natural and human-like the responses feel.
- BERTScore (30%):  
    - Evaluates semantic similarity between chatbot replies and therapist responses. Split equally between F1, Recall and Precision
- Time taken (10%)
    - Time taken to generate a response, a shorter time optimizes user experience 

|Metrics             |GPT  |Llama|LuminAI|
|--------------------|:---:|:---:|:-----:|
|Empathy Score       |0.8  |0.79 |0.79   |
|Human Likeness      |0.27 |0.45 |0.5    |
|BERTScore F1        |0.45 |0.48 |0.51   |
|BERTScore Recall    |0.51 |0.53 |0.55   |
|BERTScore Precision |0.41 |0.44 |0.47   |
|Time Taken          |89.65|15.85|39.42  |
|Total Score         |0.54 |0.65 |0.63   |

