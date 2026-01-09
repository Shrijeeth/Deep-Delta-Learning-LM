import os

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from config import get_settings
from data import WikiDataModule
from v1.model import DeepLatentConfig, DeepLatentGPT


def train():
    L.seed_everything(1337)
    torch.set_float32_matmul_precision("high")

    settings = get_settings()

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
        model_name=settings.TOKENIZER_NAME,
        batch_size=settings.BATCH_SIZE,
        max_length=settings.BLOCK_SIZE,
    )

    # 4. Model Setup
    config = DeepLatentConfig(
        vocab_size=dm.tokenizer.vocab_size,
        block_size=settings.BLOCK_SIZE,
        n_layer=8,  # Deeper is better for Latent refinement
        n_head=8,
        n_embd=384,
        dropout=0.1,
        bias=False,
        beta_init=-2.0,  # Crucial for stable start
        lr=settings.LR,
    )

    model = DeepLatentGPT(config)

    # 5. Trainer
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=settings.MAX_EPOCHS,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        max_time={"hours": settings.MAX_TRAINING_HOURS},
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    print("Starting DeepLatentGPT Training...")
    if (
        settings.IS_RESUME
        and settings.CHECKPOINT_PATH
        and os.path.exists(settings.CHECKPOINT_PATH)
    ):
        trainer.fit(model, datamodule=dm, ckpt_path=settings.CHECKPOINT_PATH)
    else:
        trainer.fit(model, datamodule=dm)
