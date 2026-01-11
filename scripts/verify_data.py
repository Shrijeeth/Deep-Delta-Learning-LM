"""
Quick script to verify data preparation is correct.
Checks that input_ids and labels are properly aligned.
"""

import sys
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import get_settings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
settings = get_settings()

# Load only 100 samples
print("Loading 100 samples from TinyStories...")
dataset = load_dataset("roneneldan/TinyStories", split="train[:100]")

tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Tokenize (matching data.py approach)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=settings.DATA_LENGTH,
        padding=False,
    )


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# Filter short sequences
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 10)


# Create custom collator that properly shifts labels for causal LM
def collate_fn(examples):
    # Pad sequences
    input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    # Create shifted labels: labels[i] = input_ids[i+1]
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]  # Shift left
    labels[:, -1] = -100  # Last position has no next token

    # Also set padding positions to -100
    labels[labels == tokenizer.pad_token_id] = -100

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


dataloader = DataLoader(tokenized_dataset, batch_size=2, collate_fn=collate_fn)

# Get one batch
batch = next(iter(dataloader))

print("=" * 60)
print("DATA VERIFICATION")
print("=" * 60)

print(f"\nBatch keys: {batch.keys()}")
print(f"Input shape: {batch['input_ids'].shape}")
print(f"Labels shape: {batch['labels'].shape}")

# Check first example
input_ids = batch["input_ids"][0]
labels = batch["labels"][0]

print("\n--- First Example ---")
print(f"Input IDs (first 10): {input_ids[:10].tolist()}")
print(f"Labels (first 10): {labels[:10].tolist()}")

# Verify alignment: labels[i] should equal input_ids[i+1]
print("\n--- Alignment Check ---")
print("Checking if labels are shifted (labels[i] should equal input_ids[i+1]):")
for i in range(min(5, len(input_ids) - 1)):
    expected_label = input_ids[i + 1].item()
    actual_label = labels[i].item()
    match = "✓" if expected_label == actual_label else "✗"
    print(
        f"Position {i}: input[{i + 1}]={expected_label}, label[{i}]={actual_label} {match}"
    )

print("\nLast few positions (should have -100 for padding):")
print(f"Labels[-5:]: {labels[-5:].tolist()}")
print(f"Input[-5:]: {input_ids[-5:].tolist()}")

# Decode to see actual text
print("\n--- Decoded Text ---")
# Filter out -100 (padding/ignore tokens)
valid_input = input_ids[input_ids != -100]
valid_labels = labels[labels != -100]

print(f"Input text: {tokenizer.decode(valid_input[:50])}")
print(f"Label text: {tokenizer.decode(valid_labels[:50])}")

print("\n" + "=" * 60)
