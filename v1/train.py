import os

import lightning as L
import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from config import get_settings
from data import WikiDataModule
from v1.model import DeepLatentConfig, DeepLatentGPT

load_dotenv()

settings = get_settings()
BLOCK_SIZE = settings.BLOCK_SIZE
BATCH_SIZE = settings.BATCH_SIZE
MAX_EPOCHS = settings.MAX_EPOCHS
LR = settings.LR
IS_RESUME = settings.IS_RESUME
CHECKPOINT_PATH = settings.CHECKPOINT_PATH


def train():
    L.seed_everything(1337)
    torch.set_float32_matmul_precision("high")

    # 1. Logger
    wandb_key = settings.WANDB_API_KEY
    if wandb_key:
        wandb.login(key=wandb_key)
        logger = WandbLogger(project="DeepLatentGPT", name="deep-delta-v1")
    else:
        logger = None
        print("Running without WandB")

    # 2. Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints_deeplatent",
        filename="deeplatent-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True,
        monitor="val/loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 3. Data
    dm = WikiDataModule(
        model_name=settings.TOKENIZER_NAME, batch_size=BATCH_SIZE, max_length=BLOCK_SIZE
    )

    # 4. Model Setup
    config = DeepLatentConfig(
        vocab_size=dm.tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=8,  # Deeper is better for Latent refinement
        n_head=8,
        n_embd=384,
        dropout=0.1,
        bias=False,
        beta_init=-2.0,  # Crucial for stable start
        lr=LR,
    )

    model = DeepLatentGPT(config)

    # 5. Trainer
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        max_time={"days": 0, "hours": 5},
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    print("Starting DeepLatentGPT Training...")
    if IS_RESUME and CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        trainer.fit(model, datamodule=dm, ckpt_path=CHECKPOINT_PATH)
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
