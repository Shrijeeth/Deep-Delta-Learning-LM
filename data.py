import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import get_settings


class WikiDataModule(L.LightningDataModule):
    def __init__(self, model_name="gpt2", batch_size=32, max_length=128):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Performance Tip: Set num_workers to your CPU count to load data faster.
        self.num_workers = get_settings().NUM_WORKERS

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # The fix we discussed earlier
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def collate_fn(self, examples):
        """Custom collator that properly shifts labels for causal LM"""
        # Pad sequences
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # Create shifted labels: labels[i] = input_ids[i+1]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # Shift left
        labels[:, -1] = -100  # Last position has no next token

        # Also set padding positions to -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def prepare_data(self):
        self.dataset = load_dataset("roneneldan/TinyStories")

    def setup(self, stage=None):
        # 1. Load raw data
        dataset = load_dataset("roneneldan/TinyStories")

        # 2. Define the tokenizer logic
        def tokenize_function(examples):
            # Tokenize with truncation to max_length
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

        # 3. Apply tokenization (Map)
        # We remove the 'text' column because the model only needs numbers (input_ids).
        tokenized_datasets = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        # 4. Filter out sequences that are too short
        def filter_short(example):
            return len(example["input_ids"]) > 10

        tokenized_datasets = tokenized_datasets.filter(filter_short)

        # 5. Split for training phases
        if stage == "fit" or stage is None:
            self.train_dataset = tokenized_datasets["train"]
            self.val_dataset = tokenized_datasets["validation"]

        if stage == "test":
            self.test_dataset = tokenized_datasets["validation"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Always shuffle training data!
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # Custom collator with proper label shifting
            pin_memory=True,  # Speed boost for data transfer to GPU
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # Custom collator with proper label shifting
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # Custom collator with proper label shifting
            pin_memory=True,
        )


class CustomPretrainingDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name="gpt2",
        dataset="roneneldan/TinyStories",
        subset=None,
        batch_size=32,
        max_length=128,
        text_column="text",
        column_preprocessing_function=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.text_column = text_column
        self.column_preprocessing_function = column_preprocessing_function

        # Performance Tip: Set num_workers to your CPU count to load data faster.
        self.num_workers = get_settings().NUM_WORKERS

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # The fix we discussed earlier
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def collate_fn(self, examples):
        """Custom collator that properly shifts labels for causal LM"""
        # Pad sequences
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # Create shifted labels: labels[i] = input_ids[i+1]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # Shift left
        labels[:, -1] = -100  # Last position has no next token

        # Also set padding positions to -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def prepare_data(self):
        self.dataset = load_dataset(self.dataset, self.subset)

    def setup(self, stage=None):
        # 1. Load raw data
        dataset = self.dataset

        if self.column_preprocessing_function:
            preprocessed_dataset = dataset.map(
                self.column_preprocessing_function,
                batched=True,
            )
        else:
            preprocessed_dataset = dataset

        # 2. Define the tokenizer logic
        def tokenize_function(examples):
            # Tokenize with truncation to max_length
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

        # 3. Apply tokenization (Map)
        # We remove the 'text' column because the model only needs numbers (input_ids).
        tokenized_datasets = preprocessed_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        # 4. Filter out sequences that are too short
        def filter_short(example):
            return len(example["input_ids"]) > 10

        tokenized_datasets = tokenized_datasets.filter(filter_short)

        # 5. Split for training phases
        if stage == "fit" or stage is None:
            self.train_dataset = tokenized_datasets["train"]
            self.val_dataset = (
                tokenized_datasets["validation"]
                if "validation" in tokenized_datasets
                else tokenized_datasets["train"].select(range(1000))
            )

        if stage == "test":
            self.test_dataset = (
                tokenized_datasets["validation"]
                if "validation" in tokenized_datasets
                else tokenized_datasets["train"].select(range(1000))
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Always shuffle training data!
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # Custom collator with proper label shifting
            pin_memory=True,  # Speed boost for data transfer to GPU
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # Custom collator with proper label shifting
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # Custom collator with proper label shifting
            pin_memory=True,
        )
