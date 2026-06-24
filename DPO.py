# ============================================================
# MINIMAL DPO TRAINING SCRIPT — FOR LEARNING
# ============================================================
# This script walks through DPO training step by step.
# Model: phi-2 (small, fast, good for experimentation)
# Dataset: Anthropic/hh-rlhf (real human preference data)
# Library: TRL (Hugging Face), which handles DPO loss for you
#
# Install requirements:
#   pip install trl transformers datasets peft accelerate bitsandbytes
# ============================================================

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# ============================================================
# STEP 1 — CONFIG
# ============================================================

MODEL_NAME = "microsoft/phi-2"   # small model, good for experimentation
                                  # swap to "mistralai/Mistral-7B-Instruct-v0.2"
                                  # if you want something bigger

BETA = 0.1          # how much to penalize drift from reference model
                    # low beta = more freedom to change
                    # high beta = stay close to SFT model

MAX_LENGTH = 512    # max token length for prompt + response
BATCH_SIZE = 2      # keep small if VRAM is tight
GRAD_ACCUM = 4      # effective batch size = BATCH_SIZE * GRAD_ACCUM = 8
LR = 1e-5           # learning rate
EPOCHS = 1          # 1 epoch is enough to see training dynamics

OUTPUT_DIR = "./dpo_output"


# ============================================================
# STEP 2 — LOAD DATASET
# ============================================================
# Anthropic/hh-rlhf is a real human preference dataset
# Each row has:
#   "chosen"  — the response humans preferred
#   "rejected" — the response humans rejected
#
# This is exactly the (x, y_w, y_l) triplet DPO needs
# except here prompt is embedded in the chosen/rejected text

print("Loading dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")  # 2000 rows for speed
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:200]")

# The hh-rlhf dataset has chosen/rejected as full conversations
# DPOTrainer expects a "prompt" column too
# Let's extract the human turn as the prompt

def extract_prompt(example):
    # The chosen response starts with the conversation
    # We extract everything up to the last "Assistant:" as the prompt
    chosen = example["chosen"]
    # Find the last Assistant turn
    last_human = chosen.rfind("\n\nHuman:")
    prompt = chosen[:last_human] if last_human != -1 else ""
    return {
        "prompt": prompt,
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

dataset = dataset.map(extract_prompt)
eval_dataset = eval_dataset.map(extract_prompt)

print(f"Dataset size: {len(dataset)}")
print(f"Sample prompt: {dataset[0]['prompt'][:200]}")
print(f"Sample chosen: {dataset[0]['chosen'][:200]}")
print(f"Sample rejected: {dataset[0]['rejected'][:200]}")


# ============================================================
# STEP 3 — LOAD MODEL IN 4BIT (QLoRA style, VRAM efficient)
# ============================================================
# We load in 4bit to save VRAM
# Then add LoRA adapters on top for efficient training
# This is the same QLoRA setup you already know from your NL-FOL project

print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # phi-2 uses fp16
    bnb_4bit_use_double_quant=True
)

# Policy model — this will be trained
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True    # needed for phi-2
)
model = prepare_model_for_kbit_training(model)

# Add LoRA adapters
# DPO trains the LoRA adapter weights, not the full model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],   # attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Reference model — this is the frozen SFT model
# TRL handles this automatically when you pass model to DPOTrainer
# It creates an internal copy and freezes it
# You don't need to manually load a second model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# STEP 4 — DPO TRAINING CONFIG
# ============================================================
# This is where you set beta and all training hyperparameters

training_args = DPOConfig(
    beta=BETA,                          # KL penalty strength
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,                   # print loss every 10 steps
    eval_steps=100,                     # evaluate every 100 steps
    save_steps=200,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_length=MAX_LENGTH,
    max_prompt_length=256,              # prompt gets at most 256 tokens
    report_to="none",                   # set to "wandb" if you use wandb
)


# ============================================================
# STEP 5 — INITIALIZE DPO TRAINER
# ============================================================
# TRL's DPOTrainer handles:
#   - Computing log π_θ(y_w|x) and log π_θ(y_l|x)
#   - Computing log π_ref(y_w|x) and log π_ref(y_l|x)
#   - Computing implicit rewards
#   - Computing DPO loss = -log σ(reward_w - reward_l)
#   - Backprop and update
# You don't implement any of this manually

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # ref_model=None means TRL auto creates reference from model
    # before any training updates — exactly the frozen SFT model concept
)


# ============================================================
# STEP 6 — TRAIN
# ============================================================
# Watch the logs as it trains. Key metrics to observe:
#
# rewards/chosen   — implicit reward for winning responses
#                    should INCREASE over training
#
# rewards/rejected — implicit reward for losing responses
#                    should DECREASE over training
#
# rewards/margins  — difference between chosen and rejected reward
#                    should INCREASE over training (model getting better at separating them)
#
# rewards/accuracies — how often model correctly prefers chosen over rejected
#                      should INCREASE toward 1.0
#
# loss             — DPO loss, should DECREASE

print("Starting DPO training...")
print("Watch these metrics:")
print("  rewards/chosen    → should increase")
print("  rewards/rejected  → should decrease")
print("  rewards/margins   → should increase")
print("  rewards/accuracies → should increase toward 1.0")
print("  loss              → should decrease")
print()

trainer.train()


# ============================================================
# STEP 7 — SAVE
# ============================================================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")


# ============================================================
# STEP 8 — QUICK INFERENCE TEST
# ============================================================
# Generate a response from the DPO trained model
# Compare to what the base phi-2 would say

from peft import PeftModel

print("\n--- Testing DPO trained model ---")

test_prompt = "\n\nHuman: Can you help me write a polite email declining a meeting?\n\nAssistant:"

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"DPO model response:\n{response}")


# ============================================================
# WHAT TO OBSERVE DURING YOUR EXPERIMENT
# ============================================================
#
# 1. rewards/margins increasing = model learning to separate
#    good responses from bad ones. This is the core DPO signal.
#
# 2. rewards/accuracies → watch this go from ~0.5 (random)
#    toward higher values as training progresses
#
# 3. If loss goes to 0 too fast = overfitting on small dataset
#    normal with 2000 examples, increase dataset size for real runs
#
# 4. Compare outputs before and after training on same prompts
#    to see qualitative difference in response quality
#
# ============================================================
