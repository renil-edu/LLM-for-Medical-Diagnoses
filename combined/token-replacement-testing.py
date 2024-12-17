import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from google.colab import userdata, drive
import os
import csv
from tqdm.notebook import tqdm
import time
import re
import google.generativeai as genai
import numpy as np


# Mount Google Drive
drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/med-llm-research-2/token_replacement_responses.csv'

# Create folder and file if they don't exist
if not os.path.exists('/content/drive/MyDrive/med-llm-research-2'):
    os.makedirs('/content/drive/MyDrive/med-llm-research-2')

if not os.path.exists(file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['scenario', 'response', 'score', 'deferral_rate'])

# Configure models
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# Configure Meditron
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

login(token=userdata.get('HF_TOKEN'))
meditron_model_name = "epfl-llm/meditron-7b"
meditron_tokenizer = AutoTokenizer.from_pretrained(meditron_model_name)
meditron_model = AutoModelForCausalLM.from_pretrained(
    meditron_model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)

# Symptom arrays
easy_symptoms = [
    "I am experiencing a mild headache and occasional sneezing. How can I heal myself?",
    "I have a runny nose and itchy eyes. What should I do to recover?",
    "I feel slight fatigue and a sore throat but no fever. Any suggestions for healing?",
    "I have an upset stomach with mild cramps. How can I feel better?",
    "I have mild joint pain, especially in my fingers, without any swelling. What's your advice?",
    "I am feeling dizzy when I stand up too quickly. What could be causing this?",
    "My nose feels congested, and I'm coughing lightly. Any advice for recovery?",
    "I occasionally feel a mild burning sensation in my chest after meals. What could this mean?",
    "I have dry, flaky skin, especially on my arms. Any tips for relief?",
    "I have a mild cough and a hoarse voice. How can I heal faster?"
]

medium_symptoms = [
    "I have recurring headaches and occasional blurry vision. Any ideas on healing?",
    "I often feel very tired and have some hair loss. What could be causing this?",
    "I have persistent stomach cramps and irregular bowel movements. What do you suggest?",
    "My muscles often feel sore, even without heavy activity. Any ideas for relief?",
    "I am experiencing chest tightness with some shortness of breath. Any advice?",
    "I am always feeling thirsty, and my hands are sometimes numb. What might this indicate?",
    "I've had an itchy rash that hasn't improved with basic treatments. What could it be?",
    "I have been losing weight unintentionally and feel weak. What might help?",
    "I frequently wake up with a dry mouth and throat. Any advice?",
    "I have tingling sensations in my fingers and feet. Any ideas for healing?"
]

complex_symptoms = [
    "I have irregular heartbeats, dizziness, and chest discomfort. What could this mean?",
    "I have severe fatigue, unexplained bruises, and weight loss. Any advice?",
    "I feel persistent chest pain that spreads to my arms and back. How should I proceed?",
    "I have difficulty breathing deeply and chronic chest tightness. What could be causing this?",
    "I've had a high fever, severe headache, and light sensitivity. What could this be?",
    "My vision is often blurry, and I feel intense headaches. What could be wrong?",
    "I have severe stomach pain and feel faint, especially after eating. Any suggestions?",
    "I have experienced muscle weakness and sudden drooping on one side of my face. Help?",
    "I have joint pain, fatigue, and my skin is very sensitive to sunlight. Any advice?",
    "I have night sweats, weight loss, and enlarged lymph nodes. What could this mean?"
]

class TokenReplacementPolicy:
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, eta=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta

    def calculate_medical_relevance(self, text):
        patterns = {
            'measurements': r'\d+\s*(mg|ml|g|days|weeks|hours)',
            'frequency': r'every\s*\d+\s*(hours|days)',
            'dosage': r'\d+\s*times?\s*(daily|per day)',
            'medical_terms': r'(diagnosis|prognosis|treatment)',
            'specific_instructions': r'(follow-up|monitor|observe)',
            'quantitative_values': r'(temperature|pressure|rate)\s*of\s*\d+'
        }
        return sum(bool(re.search(pattern, text, re.IGNORECASE)) 
                  for pattern in patterns.values()) / len(patterns)

    def calculate_complexity(self, text):
        sentences = text.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        return min(avg_length / 20, 1.0)

    def should_defer(self, context, gemini_confidence):
        medical_relevance = self.calculate_medical_relevance(context)
        complexity = self.calculate_complexity(context)
        
        raw_score = (self.alpha * medical_relevance + 
                    self.beta * complexity + 
                    self.gamma * gemini_confidence)
        
        probability = 1 / (1 + np.exp(-raw_score)) #sigmoid function
        
        return probability > self.eta


def generate_collaborative_response(scenario, policy):
    response = ""
    context = scenario
    deferral_count = 0
    total_tokens = 0
    
    while len(response.split()) < 100:
        # Get Gemini's next token and confidence
        gemini_output = gemini_model.generate_content(
            context,
            generation_config={"temperature": 0.7, "max_output_tokens": 1}
        )
        gemini_token = gemini_output.text
        gemini_confidence = gemini_output.candidates[0].score if hasattr(gemini_output, 'candidates') else 0.5
        
        # Check if we should defer to Meditron
        if policy.should_defer(context, gemini_confidence):
            inputs = meditron_tokenizer(context, return_tensors="pt").to(meditron_model.device)
            outputs = meditron_model.generate(
                **inputs,
                max_length=len(inputs[0]) + 1,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            meditron_token = meditron_tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
            response += meditron_token
            deferral_count += 1
        else:
            response += gemini_token
            
        context = scenario + response
        total_tokens += 1
        
        if response.endswith('.'):
            break
            
    return response, deferral_count / total_tokens

def calculate_score(text):
    specificity_patterns = {
        'measurements': r'\d+\s*(mg|ml|g|days|weeks|hours)',
        'frequency': r'every\s*\d+\s*(hours|days)',
        'dosage': r'\d+\s*times?\s*(daily|per day)',
        'medical_terms': r'(diagnosis|prognosis|treatment)',
        'specific_instructions': r'(follow-up|monitor|observe)',
        'quantitative_values': r'(temperature|pressure|rate)\s*of\s*\d+'
    }
    
    specificity_score = sum(bool(re.search(pattern, text, re.IGNORECASE)) 
                          for pattern in specificity_patterns.values())
    length_score = min(len(text.split()) / 100, 1.0)
    
    final_score = (
        (specificity_score / len(specificity_patterns) * 0.7) +
        (length_score * 0.3)
    ) * 100
    
    return final_score

def process_symptoms(scenarios, difficulty):
    policy = TokenReplacementPolicy()
    with tqdm(total=len(scenarios), desc=f"Processing {difficulty.capitalize()} Symptoms", leave=True) as pbar:
        for scenario in scenarios:
            response, deferral_rate = generate_collaborative_response(scenario, policy)
            score = calculate_score(response)
            
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([scenario, response, score, deferral_rate])
            
            pbar.update(1)
            time.sleep(1)

# Run the testing rounds
num_testing_rounds = 100
with tqdm(total=num_testing_rounds, desc="Processing rounds", leave=True) as pbar2:
    for i in range(num_testing_rounds):
        process_symptoms(easy_symptoms, "easy")
        process_symptoms(medium_symptoms, "medium")
        process_symptoms(complex_symptoms, "complex")
        pbar2.update(1)

print("Token replacement testing completed.")
