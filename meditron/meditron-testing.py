import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from google.colab import userdata, drive
import os
import csv
from tqdm.notebook import tqdm
import time

# Mount Google Drive
drive.mount('/content/drive')

# Define path to the file
file_path = '/content/drive/MyDrive/med-llm-research-2/meditron_llm_full_diagnosis_responses.csv'

# Create folder and file if they don't exist
if not os.path.exists('/content/drive/MyDrive/med-llm-research-2'):
    os.makedirs('/content/drive/MyDrive/med-llm-research-2')

# Initialize the CSV file with headers if it does not exist
if not os.path.exists(file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['confidence', 'diagnosis'])
    print("File initialized.")
else:
    print("File already exists. Ready for appending data.")

# Configure for 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Log in to Hugging Face
login(token=userdata.get('HF_TOKEN'))

# Load model and tokenizer
model_name = "epfl-llm/meditron-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )
else:
    print("GPU not available. Loading on CPU.")
    model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_medical_response(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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

# Dictionary to store responses
responses = {
    "easy": [],
    "medium": [],
    "complex": []
}

def process_symptoms(scenarios, difficulty):
    with tqdm(total=len(scenarios), desc=f"Processing {difficulty.capitalize()} Symptoms", leave=True) as pbar:
        for scenario in scenarios:
            prompt = f"""Imagine you're diagnosing a patient. They report the following symptoms: {scenario}.
Based on your assessment, what would your confidence level (1-10) be in providing a diagnosis, where 1 is the least confident and 10 is the most confident?
Please then provide your diagnosis, and steps you would have the patient take to cure their illness.
Please respond with an integer between 1 and 10, a comma, a space, and then your diagnosis and instructions for the patient to cure."""

            response = generate_medical_response(prompt)
            
            try:
                parts = response.split(", ", 1)
                if len(parts) == 2:
                    confidence = parts[0]
                    diagnosis = parts[1]
                    responses[difficulty].append((int(confidence), diagnosis))
                    
                    print(f"\nConfidence Score: {confidence}")
                    print(f"Diagnosis: {diagnosis}\n")
                    
                    with open(file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([confidence, diagnosis])
                else:
                    print(f"Invalid response format: {response}")
            except ValueError:
                print(f"Error parsing response: {response}")
            
            pbar.update(1)
            time.sleep(4)  # Respect rate limits

# Run the testing rounds
num_testing_rounds = 100
with tqdm(total=num_testing_rounds, desc="Processing rounds", leave=True) as pbar2:
    for i in range(num_testing_rounds):
        process_symptoms(easy_symptoms, "easy")
        process_symptoms(medium_symptoms, "medium")
        process_symptoms(complex_symptoms, "complex")
        pbar2.update(1)
