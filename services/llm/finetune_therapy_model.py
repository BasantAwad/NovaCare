"""
Fine-tuning Script for NovaBot Mental Health Therapy Model
=========================================================
This script fine-tunes a base LLM (e.g., Llama-3-8B) using Parameter-Efficient 
Fine-Tuning (LoRA) to prevent overfitting and reduce memory usage.

DATASETS:
1. Hugging Face Datasets (Downloaded Automatically):
   - Amod/mental_health_counseling_conversations

2. Kaggle Datasets (Downloaded Automatically via kagglehub):
   - melissamonfared/mental-health-counseling-conversations-k
   - zuhairhasanshaik/datacsv
   - bhavikjikadara/mental-health-dataset

REQUIREMENTS:
pip install transformers peft trl datasets torch wandb kagglehub[pandas-datasets]
"""

import os
import glob
import pandas as pd
import torch
from dotenv import load_dotenv
from datasets import load_dataset, Dataset, concatenate_datasets
import kagglehub
from kagglehub import KaggleDatasetAdapter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# If you are in Colab, PASTE YOUR HUGGING FACE TOKEN HERE inside the quotes:
# Example: HF_TOKEN = "hf_YourTokenHere123"
HF_TOKEN = ""  # Your token from .env

# Fallback to .env if running locally and the token above is empty
if not HF_TOKEN:
    load_dotenv(".env", override=True)
    HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

if HF_TOKEN:
    print("[*] Logging into Hugging Face...")
    login(token=HF_TOKEN)
else:
    print("[!] WARNING: No HF_TOKEN provided. You may get 401 Unauthorized errors.")

BASE_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"  # Ungated version of Llama 3
OUTPUT_DIR = "./therapy_model_lora"
LOCAL_DATA_DIR = "./data/raw"

# Hyperparameters (tuned to prevent overfitting)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4

# ---------------------------------------------------------------------------
# 1. Data Loading & Preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess_data():
    print("[*] Loading datasets...")
    datasets_list = []

    # A. Hugging Face Datasets (Automatic API Download)
    try:
        print("    -> Downloading 'Amod/mental_health_counseling_conversations' from HF...")
        hf_ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
        
        # Standardize format to 'instruction', 'input' (optional), and 'output'
        def format_amod(example):
            return {
                "text": f"<|system|>\nYou are an empathetic mental health AI therapist.<|end|>\n<|user|>\n{example['Context']}<|end|>\n<|assistant|>\n{example['Response']}<|end|>"
            }
        
        hf_ds = hf_ds.map(format_amod, remove_columns=hf_ds.column_names)
        datasets_list.append(hf_ds)
        print(f"    [OK] Loaded {len(hf_ds)} HF examples.")
    except Exception as e:
        print(f"    [FAIL] Could not load HF dataset: {e}")

    # B. Kaggle Datasets (Automatic API Download via kagglehub)
    kaggle_datasets = [
        "melissamonfared/mental-health-counseling-conversations-k",
        "zuhairhasanshaik/datacsv",
        "bhavikjikadara/mental-health-dataset"
    ]
    
    for k_ds in kaggle_datasets:
        try:
            print(f"    -> Downloading '{k_ds}' from Kaggle...")
            df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, k_ds, "")
            
            # Dynamically identify the Question and Answer columns
            q_col = next((col for col in df.columns if col.lower() in ['question', 'patient', 'context', 'user', 'text']), None)
            a_col = next((col for col in df.columns if col.lower() in ['answer', 'therapist', 'response', 'assistant', 'label']), None)
            
            # If standard columns exist, process them
            if q_col and a_col and q_col != a_col:
                # Filter out NA rows
                df = df.dropna(subset=[q_col, a_col])
                df['text'] = df.apply(lambda row: f"<|system|>\nYou are an empathetic mental health AI therapist.<|end|>\n<|user|>\n{row[q_col]}<|end|>\n<|assistant|>\n{row[a_col]}<|end|>", axis=1)
                
                local_ds = Dataset.from_pandas(df[['text']])
                datasets_list.append(local_ds)
                print(f"    [OK] Loaded {len(df)} examples from {k_ds}.")
            else:
                print(f"    [WARN] Could not automatically identify Q/A columns for {k_ds}. Available: {list(df.columns)}")
        except Exception as e:
            print(f"    [FAIL] Could not load Kaggle dataset {k_ds}: {e}")

    if not datasets_list:
        raise ValueError("No datasets loaded! Please check internet connection or local files.")

    # Combine all datasets
    combined_ds = concatenate_datasets(datasets_list)
    
    # Shuffle and create Train/Validation split (90/10) to monitor overfitting
    print("[*] Shuffling and splitting data (90% Train, 10% Eval)...")
    combined_ds = combined_ds.shuffle(seed=42)
    split_ds = combined_ds.train_test_split(test_size=0.1, seed=42)
    
    return split_ds['train'], split_ds['test']


# ---------------------------------------------------------------------------
# 2. Model & LoRA Setup
# ---------------------------------------------------------------------------
def setup_model_and_tokenizer():
    print(f"[*] Loading Tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, token=HF_TOKEN)
    
    # Llama 3 requires a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[*] Loading Base Model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16, # Use bf16 if supported by your GPU
        token=HF_TOKEN
    )
    
    # Prepare for LoRA to prevent catastrophic forgetting and overfitting
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=LORA_DROPOUT, # Dropout helps prevent overfitting
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------
def main():
    train_data, val_data = load_and_preprocess_data()
    model, tokenizer = setup_model_and_tokenizer()
    
    print("[*] Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        optim="adamw_torch",
        lr_scheduler_type="cosine",    # Smooth decay prevents overfitting
        weight_decay=0.01,             # L2 regularization to prevent overfitting
        warmup_ratio=0.05,
        fp16=True,                     # Use bf16=True for Ampere GPUs
        report_to="none",              # Change to 'wandb' to track metrics
        load_best_model_at_end=True,   # Keeps the model that performed best on Val set
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Early stopping callback: stops training if validation loss doesn't improve for 3 checks
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[early_stopping],
    )
    
    print("[*] Starting Fine-Tuning...")
    trainer.train()
    
    print(f"[*] Training complete. Saving final adapter to {OUTPUT_DIR}/final...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print("[*] Done!")

if __name__ == "__main__":
    main()
