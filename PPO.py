# ============================================================
# MINIMAL PPO RLHF TRAINING SCRIPT — FOR LEARNING
# ============================================================
# This script walks through PPO based RLHF step by step.
# Compare this with dpo_minimal.py to see the complexity difference.
#
# Model: phi-2 (policy + reference + reward + value = 4 models)
# Dataset: Anthropic/hh-rlhf (we use chosen responses for reward model training)
# Library: TRL (Hugging Face PPOTrainer)
#
# RLHF has THREE distinct training stages:
#   Stage 1 — SFT (we skip this, use phi-2 as our SFT model)
#   Stage 2 — Train Reward Model on preference data
#   Stage 3 — PPO: optimize policy against reward model
#
# Install requirements:
#   pip install trl transformers datasets peft accelerate bitsandbytes
# ============================================================

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import RewardTrainer, RewardConfig
import numpy as np

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "microsoft/phi-2"
BETA = 0.1                  # KL penalty coefficient (same as DPO beta)
MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR_REWARD = 1e-5            # reward model learning rate
LR_PPO = 1e-5               # policy learning rate
PPO_EPOCHS = 4              # how many PPO update steps per batch of episodes
OUTPUT_DIR_REWARD = "./reward_model_output"
OUTPUT_DIR_PPO = "./ppo_output"


# ============================================================
# LOAD TOKENIZER (shared across all models)
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"     # important for PPO generation


# ============================================================
# STAGE 1 — SFT MODEL
# ============================================================
# We treat phi-2 as our SFT model directly.
# In a real pipeline you would first fine tune phi-2 on
# instruction following data before this step.
# We skip that here to keep the script focused on RLHF.

print("=" * 60)
print("STAGE 1 — SFT MODEL")
print("Using phi-2 as our SFT model directly (skipping SFT training)")
print("=" * 60)


# ============================================================
# STAGE 2 — REWARD MODEL TRAINING
# ============================================================
# This is the stage DPO eliminates entirely.
# Here we:
#   1. Load preference data (chosen vs rejected pairs)
#   2. Train a separate model to score responses
#   3. The reward model learns: score(chosen) > score(rejected)
#
# Architecture:
#   Same backbone as LLM but final layer outputs a scalar
#   instead of vocabulary logits

print("\n" + "=" * 60)
print("STAGE 2 — REWARD MODEL TRAINING")
print("=" * 60)

# Load preference dataset
# Each row: {"chosen": full_conversation_chosen, "rejected": full_conversation_rejected}
print("Loading preference dataset...")
reward_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:3000]")
reward_eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:300]")

# Load reward model
# AutoModelForSequenceClassification outputs a scalar score per input
# num_labels=1 means one scalar output = the reward score
print("Loading reward model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,               # scalar reward output
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
reward_model = prepare_model_for_kbit_training(reward_model)

# Add LoRA to reward model
reward_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"         # sequence classification task type
)
reward_model = get_peft_model(reward_model, reward_lora_config)
reward_model.print_trainable_parameters()

# RewardTrainer handles the Bradley-Terry loss automatically:
#   Loss = -log(sigmoid(score(chosen) - score(rejected)))
# This pushes score(chosen) > score(rejected) for every pair

reward_training_args = RewardConfig(
    output_dir=OUTPUT_DIR_REWARD,
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR_REWARD,
    fp16=True,
    logging_steps=10,
    eval_steps=100,
    save_steps=200,
    max_length=MAX_LENGTH,
    report_to="none",
)

reward_trainer = RewardTrainer(
    model=reward_model,
    args=reward_training_args,
    train_dataset=reward_dataset,
    eval_dataset=reward_eval_dataset,
    tokenizer=tokenizer,
)

# Key metrics to watch during reward model training:
#   rewards/chosen   → should increase
#   rewards/rejected → should decrease
#   rewards/margins  → should increase (separation between chosen/rejected)
#   rewards/accuracies → should increase toward 1.0

print("Training reward model...")
print("Watch: rewards/margins increasing = reward model learning preferences")
reward_trainer.train()
reward_trainer.save_model(OUTPUT_DIR_REWARD)
print(f"Reward model saved to {OUTPUT_DIR_REWARD}")


# ============================================================
# STAGE 3 — PPO TRAINING
# ============================================================
# Now we use the trained reward model as our reward signal
# and optimize the policy (SFT model) using PPO.
#
# What PPO manages internally:
#   - Policy model π_θ (being trained)
#   - Reference model π_ref (frozen SFT model for KL penalty)
#   - Value network V_φ (critic, estimates expected future reward)
#   - Reward model (we pass this in, already trained above)
#
# The training loop per step:
#   1. Sample prompts from dataset
#   2. Policy generates responses (forward pass through environment)
#   3. Reward model scores each response
#   4. KL penalty computed between policy and reference model
#   5. Final reward = RM score - β × KL
#   6. PPO computes advantages using value network
#   7. Clipped PPO loss computed
#   8. Policy and value network updated

print("\n" + "=" * 60)
print("STAGE 3 — PPO TRAINING")
print("=" * 60)

# Load policy model with value head
# AutoModelForCausalLMWithValueHead adds a value network on top of the LLM
# This is the actor-critic setup we discussed — same model, two heads:
#   LM head  → outputs token probabilities (actor)
#   Value head → outputs scalar state value (critic)

print("Loading policy model with value head...")

policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
)

# Load prompts dataset for PPO
# PPO needs prompts to generate responses from
# We use the human turns from hh-rlhf as prompts

print("Loading prompts dataset...")
ppo_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")

def extract_prompt_only(example):
    # Extract just the last human turn as the prompt
    chosen = example["chosen"]
    last_assistant = chosen.rfind("\n\nAssistant:")
    prompt = chosen[:last_assistant] + "\n\nAssistant:" if last_assistant != -1 else chosen
    return {"query": prompt}

ppo_dataset = ppo_dataset.map(extract_prompt_only)
ppo_dataset = ppo_dataset.filter(lambda x: len(x["query"]) < 512)

# PPO Config
# Key params:
#   kl_penalty — how to compute KL (kl or abs or mse or full)
#   init_kl_coef — initial β for KL penalty (same concept as DPO beta)
#   cliprange — ε for PPO clipping (prevents too large policy updates)
#   vf_coef — how much to weight value loss vs policy loss

ppo_config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=LR_PPO,
    batch_size=BATCH_SIZE * GRAD_ACCUM,     # effective batch size
    mini_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    ppo_epochs=PPO_EPOCHS,                  # PPO update steps per batch
    kl_penalty="kl",                        # standard KL divergence penalty
    init_kl_coef=BETA,                      # initial KL penalty coefficient
    cliprange=0.2,                          # ε for clipping (standard PPO value)
    cliprange_value=0.2,                    # ε for value function clipping
    vf_coef=0.1,                            # value loss coefficient
    max_grad_norm=0.5,                      # gradient clipping for stability
    fp16=True,
    log_with=None,                          # set to "wandb" for logging
    output_dir=OUTPUT_DIR_PPO,
)

# Initialize PPO Trainer
# PPOTrainer internally:
#   - Keeps a frozen copy of policy as reference model
#   - Computes KL between policy and reference at each step
#   - Manages advantage computation using value head
#   - Applies clipped PPO update

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=None,         # None = auto create frozen reference from policy
    tokenizer=tokenizer,
    dataset=ppo_dataset,
    data_collator=lambda data: {
        "input_ids": [d["input_ids"] for d in data],
        "query": [d["query"] for d in data]
    }
)

# Load the trained reward model for scoring
# We use it as a black box function: (prompt, response) → scalar score
trained_reward_model = AutoModelForSequenceClassification.from_pretrained(
    OUTPUT_DIR_REWARD,
    device_map="auto",
    trust_remote_code=True
)
trained_reward_model.eval()

def get_reward_scores(prompts, responses):
    """
    Score each (prompt, response) pair using the reward model.
    Returns a list of scalar tensors — one score per response.
    This is the reward signal PPO uses.
    """
    scores = []
    for prompt, response in zip(prompts, responses):
        inputs = tokenizer(
            prompt + response,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(trained_reward_model.device)

        with torch.no_grad():
            output = trained_reward_model(**inputs)
            score = output.logits[0][0]     # scalar reward score
        scores.append(score)
    return scores


# ============================================================
# PPO TRAINING LOOP
# ============================================================
# This is the core difference from DPO.
# DPO: one forward pass per (prompt, chosen, rejected) → loss → update
# PPO: generate response → score it → compute advantage → clipped update
#
# PPO has an explicit episode collection step that DPO doesn't need.

print("Starting PPO training loop...")
print("Watch these metrics:")
print("  ppo/mean_scores      → average reward model score, should increase")
print("  ppo/mean_non_score_reward → KL penalty, watch it doesn't explode")
print("  ppo/policy/clipfrac  → fraction of clipped updates, should stay low")
print("  objective/kl         → KL from reference model, should stay bounded")
print()

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 128,
}

for epoch, batch in enumerate(ppo_trainer.dataloader):
    if epoch >= 50:     # run 50 steps for demo, remove for full training
        break

    # Step 1 — Get prompts from batch
    query_tensors = batch["input_ids"]
    queries = batch["query"]

    # Step 2 — Generate responses from policy
    # This is the "episode" — policy interacts with environment
    # Each token generation is one action in the MDP
    response_tensors = ppo_trainer.generate(
        query_tensors,
        return_prompt=False,
        **generation_kwargs
    )
    responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Step 3 — Score responses with reward model
    # This is where the reward signal comes from
    # In the grid game analogy: this is the environment giving back reward
    rewards = get_reward_scores(queries, responses)

    # Step 4 — PPO update
    # PPOTrainer handles internally:
    #   - KL penalty computation (reward = RM score - β × KL)
    #   - Value network forward pass to get V(s_t) at each token
    #   - Advantage computation A_t = G_t - V(s_t)
    #   - Clipped PPO loss computation
    #   - Policy and value network update
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    if epoch % 10 == 0:
        print(f"Step {epoch}")
        print(f"  Mean reward score : {np.mean([r.item() for r in rewards]):.4f}")
        print(f"  KL from reference : {stats.get('objective/kl', 0):.4f}")
        print(f"  Clip fraction     : {stats.get('ppo/policy/clipfrac', 0):.4f}")
        print()

# Save final PPO model
ppo_trainer.save_pretrained(OUTPUT_DIR_PPO)
print(f"PPO model saved to {OUTPUT_DIR_PPO}")


# ============================================================
# QUICK INFERENCE TEST
# ============================================================

print("\n--- Testing PPO trained model ---")

test_prompt = "\n\nHuman: Can you help me write a polite email declining a meeting?\n\nAssistant:"
inputs = tokenizer(test_prompt, return_tensors="pt").to(policy_model.pretrained_model.device)

with torch.no_grad():
    outputs = policy_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"PPO model response:\n{response}")


# ============================================================
# DPO vs PPO — SIDE BY SIDE COMPARISON
# ============================================================
#
# MODELS IN MEMORY:
#   DPO: policy + reference = 2 models
#   PPO: policy + reference + reward model + value head = 4 models
#
# TRAINING STAGES:
#   DPO: 1 stage (direct policy optimization on preference data)
#   PPO: 2 stages (reward model training → PPO loop)
#
# TRAINING LOOP:
#   DPO: for each (prompt, chosen, rejected) → loss → update
#        same as supervised fine tuning, no episode generation needed
#   PPO: for each prompt → generate response → score → compute advantage → update
#        requires explicit episode collection, much more complex
#
# STABILITY:
#   DPO: stable, behaves like supervised training
#   PPO: sensitive to hyperparameters, reward hacking is a real risk
#        cliprange, kl_coef, ppo_epochs all need careful tuning
#
# VRAM:
#   DPO: ~2x model size (policy + reference)
#   PPO: ~4x model size (policy + reference + reward + value)
#
# WHEN TO USE WHICH:
#   DPO: most alignment fine tuning tasks, easier to run, widely used now
#   PPO: when you need online learning (generating and scoring new data
#        during training), or when reward signal is from a real environment
#        not a preference dataset
#
# ============================================================
