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

# Mount Google Drive
drive.mount('/content/drive')

# Define path to the file
file_path = '/content/drive/MyDrive/med-llm-research-2/manual_merged_responses.csv'

# Create folder and file if they don't exist
if not os.path.exists('/content/drive/MyDrive/med-llm-research-2'):
    os.makedirs('/content/drive/MyDrive/med-llm-research-2')

# Initialize the CSV file with headers if it does not exist
if not os.path.exists(file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['scenario', 'gemini_output', 'meditron_output', 'merged_output', 'score'])
    print("File initialized.")
else:
    print("File already exists. Ready for appending data.")

# Configure Gemini
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

# Easy-to-diagnose symptoms array
easy_symptoms = [
    "I am experiencing a mild headache and occasional sneezing. How can I heal myself?",
    "I have a runny nose and itchy eyes. What should I do to recover?",
    "I feel slight fatigue and a sore throat but no fever. Any suggestions for healing?",
    "I have an upset stomach with mild cramps. How can I feel better?",
    "I have mild joint pain, especially in my fingers, without any swelling. What’s your advice?",
    "I am feeling dizzy when I stand up too quickly. What could be causing this?",
    "My nose feels congested, and I’m coughing lightly. Any advice for recovery?",
    "I occasionally feel a mild burning sensation in my chest after meals. What could this mean?",
    "I have dry, flaky skin, especially on my arms. Any tips for relief?",
    "I have a mild cough and a hoarse voice. How can I heal faster?"
]

# Medium-complexity symptoms array
medium_symptoms = [
    "I have recurring headaches and occasional blurry vision. Any ideas on healing?",
    "I often feel very tired and have some hair loss. What could be causing this?",
    "I have persistent stomach cramps and irregular bowel movements. What do you suggest?",
    "My muscles often feel sore, even without heavy activity. Any ideas for relief?",
    "I am experiencing chest tightness with some shortness of breath. Any advice?",
    "I am always feeling thirsty, and my hands are sometimes numb. What might this indicate?",
    "I’ve had an itchy rash that hasn’t improved with basic treatments. What could it be?",
    "I have been losing weight unintentionally and feel weak. What might help?",
    "I frequently wake up with a dry mouth and throat. Any advice?",
    "I have tingling sensations in my fingers and feet. Any ideas for healing?"
]

# Complex symptoms array
complex_symptoms = [
    "I have irregular heartbeats, dizziness, and chest discomfort. What could this mean?",
    "I have severe fatigue, unexplained bruises, and weight loss. Any advice?",
    "I feel persistent chest pain that spreads to my arms and back. How should I proceed?",
    "I have difficulty breathing deeply and chronic chest tightness. What could be causing this?",
    "I’ve had a high fever, severe headache, and light sensitivity. What could this be?",
    "My vision is often blurry, and I feel intense headaches. What could be wrong?",
    "I have severe stomach pain and feel faint, especially after eating. Any suggestions?",
    "I have experienced muscle weakness and sudden drooping on one side of my face. Help?",
    "I have joint pain, fatigue, and my skin is very sensitive to sunlight. Any advice?",
    "I have night sweats, weight loss, and enlarged lymph nodes. What could this mean?"
]

def generate_gemini_response(prompt):
    response = gemini_model.generate_content(prompt)
    return response.text

def generate_meditron_response(prompt, max_length=512):
    inputs = meditron_tokenizer(prompt, return_tensors="pt").to(meditron_model.device)
    outputs = meditron_model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=meditron_tokenizer.eos_token_id
    )
    return meditron_tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    with tqdm(total=len(scenarios), desc=f"Processing {difficulty.capitalize()} Symptoms", leave=True) as pbar:
        for scenario in scenarios:
            gemini_output = generate_gemini_response(scenario)
            meditron_output = generate_meditron_response(scenario)
            
            print(f"\nScenario: {scenario}")
            print(f"Gemini output: {gemini_output}")
            print(f"Meditron output: {meditron_output}")
            
            merged_output = input("Enter the merged output: ")
            score = calculate_score(merged_output)
            
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([scenario, gemini_output, meditron_output, merged_output, score])
            
            pbar.update(1)
            time.sleep(1)  # Short delay between iterations

# Run the testing rounds
num_testing_rounds = 1  # Adjust as needed
with tqdm(total=num_testing_rounds, desc="Processing rounds", leave=True) as pbar2:
    for i in range(num_testing_rounds):
        process_symptoms(easy_symptoms, "easy")
        process_symptoms(medium_symptoms, "medium")
        process_symptoms(complex_symptoms, "complex")
        pbar2.update(1)

print("Manual merging process completed.")
