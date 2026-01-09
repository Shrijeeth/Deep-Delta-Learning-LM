import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

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

    def prepare_data(self):
        self.dataset = load_dataset("roneneldan/TinyStories")

    def setup(self, stage=None):
        # 1. Load raw data
        dataset = load_dataset("roneneldan/TinyStories")

        # 2. Define the tokenizer logic
        def tokenize_function(examples):
            # We truncate here to ensure no sequence exceeds our max memory
            return self.tokenizer(
                examples["text"], truncation=False, max_length=self.max_length
            )

        # 3. Apply tokenization (Map)
        # We remove the 'text' column because the model only needs numbers (input_ids).
        tokenized_datasets = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        def group_texts(examples):
            concatenated = {}
            for key in examples.keys():
                concatenated[key] = sum(examples[key], [])
            return concatenated

        tokenized_datasets = tokenized_datasets.map(group_texts, batched=True)

        def split_into_blocks(examples):
            input_ids = examples["input_ids"]
            total_length = (len(input_ids) // self.max_length) * self.max_length
            input_ids = input_ids[:total_length]

            return {
                "input_ids": [
                    input_ids[i : i + self.max_length]
                    for i in range(0, total_length, self.max_length)
                ]
            }

        tokenized_datasets = tokenized_datasets.map(
            split_into_blocks,
            batched=True,
            remove_columns=tokenized_datasets["train"].column_names,
        )

        # 4. Filter out empty sequences (this is the fix!)
        def filter_empty(example):
            return len(example["input_ids"]) > 0

        tokenized_datasets = tokenized_datasets.filter(filter_empty)

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
            # This is where Dynamic Padding happens:
            collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            pin_memory=True,  # Speed boost for data transfer to GPU
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            pin_memory=True,
        )
