# Deep-Delta-Learning-LM

Implementation of a **Deep Delta Learning** language model inspired by the paper  
[*Deep Delta Learning* (Zhang et al., 2026, arXiv:2601.00417)](https://arxiv.org/abs/2601.00417).
The core idea replaces residual “add” updates with a geometric *erase–write* step that explicitly
controls **direction**, **target value**, and **gate strength** in latent space.

---

## Why Deep Delta?

Classic residual blocks do `x ← x + f(x)`, which hard-codes additive bias. Deep Delta generalizes this
by learning how much to erase and rewrite along a unit direction. For a latent state \(x \in \mathbb{R}^d\)
and a proposed residual signal \(r\):

1. **Direction** \(k(x) = \text{normalize}(\text{MLP}(x)) \in \mathbb{R}^d\)  
2. **Gate** \(\beta(x) = 2\sigma(W_\beta x) \in [0,2]\) (0 = keep, 1 = replace, 2 = reflect)  
3. **Target value** \(v(r) = W_v r \in \mathbb{R}\)

Projection of the current state onto the direction:
\[
 \langle x, k \rangle = x^\top k
\]

Geometric delta and update:
\[
\Delta = \beta \, k \, (v - \langle x, k \rangle), \qquad x_{\text{new}} = x + \Delta
\]

This “erase–write” rule makes the residual path learn *where* and *how strongly* to update, improving
optimization stability and expressivity—especially for deeper stacks.

---

## Model overview (v1)

- **Backbone:** GPT-style decoder stack (causal self-attention + MLP) implemented in PyTorch Lightning.  
- **Delta blocks:** Each attention and MLP sub-block is wrapped with `DeltaLatentUpdate`, replacing
  residual addition by the geometric update above.  
- **Config:** `DeepLatentConfig` in `v1/model.py` controls depth (`n_layer=8`), heads (`n_head=8`),
  embedding dim (`n_embd=384`), dropout, and gate initialization (`beta_init=-2` keeps gates nearly
  closed at start).  
- **Weight tying:** Token embedding and LM head are tied.  
- **Optimization:** AdamW with decoupled weight decay groups (0.1 weight decay on matrix params,
  none on biases/LayerNorm), betas (0.9, 0.95); gradient clipping handled by the Trainer; LR comes
  from `config.py`. Perplexity is tracked during train/val.  
- **Data:** TinyStories via Hugging Face `datasets`, tokenized with `AutoTokenizer` (defaults to `gpt2`).
  Dynamic padding via `DataCollatorForLanguageModeling`.  
- **Generation:** `DeepLatentGPT.generate` supports temperature, top-k, and repetition penalty.  
- **Logging/ckpts:** Optional Weights & Biases logging, Lightning checkpoints kept in
  `checkpoints_deeplatent/`.

Key code: `v1/model.py` (architecture + delta math), `v1/train.py` (orchestration), `data.py`
(TinyStories datamodule), `config.py` (env-driven settings).

---

## Math intuition vs. standard residuals

- **Residual (baseline):** \(x_{l+1} = x_l + f(x_l)\) — always additive, direction tied to \(f\).
- **Deep Delta:** \(x_{l+1} = x_l + \beta k (v - \langle x_l, k \rangle)\) — separates *where* to
  change (direction \(k\)), *how far* (difference to target \(v\)), and *how much* (gate \(\beta\)).
- **Gate init:** Biasing \(\beta\) toward 0 (negative logits) keeps early training near identity,
  stabilizing deep stacks before learning stronger edits.

---

## Setup

```bash
pip install -r requirements.txt
# or
make install
```

Environment variables (see `config.py`):

- `WANDB_API_KEY` (optional) — enable WandB logging
- `TOKENIZER_NAME` (default: gpt2)
- `DATA_LENGTH` (default 128) — tokenizer max length before blocking
- `BLOCK_SIZE` (sequence length, default 512)
- `BATCH_SIZE` (default 16)
- `MAX_EPOCHS` (default 3)
- `LR` (default 3e-4)
- `NUM_WORKERS` (default 6) — dataloader workers
- `IS_RESUME`, `CHECKPOINT_PATH` for resuming training
- `MAX_TRAINING_HOURS` (default 5) — Lightning `max_time` limit
- `AWS_ENABLED` (default False) — enable post-train upload
- `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_BUCKET_NAME`, `S3_MODEL_PATH` — S3-compatible upload target for `checkpoints_deeplatent/last.ckpt`

---

## Training

```bash
# Run the CLI entrypoint (loads .env and dispatches by version/mode)
python main.py --version v1 --mode train
```

What happens:

1. Seeds everything, sets matmul precision high.
2. Builds TinyStories datamodule with dynamic padding.
3. Instantiates `DeepLatentGPT` with the delta blocks.
4. Runs Lightning Trainer with checkpointing, LR monitor, optional WandB logger, and a wall-clock cap (`max_time`) set via `MAX_TRAINING_HOURS`.

To resume:

```bash
IS_RESUME=true CHECKPOINT_PATH=checkpoints_deeplatent/last.ckpt python main.py --version v1
```

---

## Inference (generation)

`v1/inference.py` is a lightweight helper; typical flow:

```python
from v1.inference import run_inference

# Adjust defaults inside v1/inference.py (PROMPT, CHECKPOINT_PATH, MAX_NEW_TOKENS,
# TEMPERATURE, TOP_K, REPETITION_PENALTY) or set CHECKPOINT_PATH via .env.
run_inference()
```

CLI inference (uses defaults in `v1/inference.py` and `CHECKPOINT_PATH` from .env):

```bash
python main.py --version v1 --mode inference
```

What happens:

1. Loads settings from `.env` (tokenizer name, block size, checkpoint path, etc.).
2. Builds `DeepLatentGPT` with `DeepLatentConfig` using `BLOCK_SIZE` from settings and tokenizer vocab.
3. Loads the checkpoint at `CHECKPOINT_PATH` (errors if missing).
4. Generates with temperature, top-k, and repetition penalty settings defined in `v1/inference.py`.

---

## Project layout

- `v1/model.py` — Deep Delta blocks, GPT backbone, Lightning module
- `v1/train.py` — training script + Trainer setup
- `main.py` — CLI entrypoint that loads `.env` and routes to a versioned trainer
- `v1/inference.py` — checkpoint-loading inference helper (temperature/top-k/repetition penalty)
- `data.py` — TinyStories datamodule (tokenize, block, pad)
- `config.py` — environment-driven hyperparameters
- `Makefile` — install/format/lint helpers

---

## References

- Zhang, Yifan; Liu, Yifeng; Wang, Mengdi; Gu, Quanquan. *Deep Delta Learning*. arXiv:2601.00417, 2026.
  [arxiv.org/abs/2601.00417](https://arxiv.org/abs/2601.00417)
- Paszke, Adam; Gross, Sam; Massa, Francisco; Lerer, Adam; Bradbury, James; Chanan, Gregory; Killeen, Trevor; Lin, Zeming; Gimelshein, Natalia; Antiga, Luca; et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019. arXiv:1912.01703. [arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703)
- PyTorch Lightning project documentation. [lightning.ai/docs/pytorch/stable](https://lightning.ai/docs/pytorch/stable/)
- Eldan, Ronen; Li, Yuanzhi. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759, 2023. [arxiv.org/abs/2305.07759](https://arxiv.org/abs/2305.07759)
- Radford, Alec; Wu, Jeffrey; Child, Rewon; Luan, David; Amodei, Dario; Sutskever, Ilya. *Language Models are Unsupervised Multitask Learners* (GPT-2 Technical Report), 2019. [cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## VM Deployment

For deploying and running training on a remote Ubuntu VM, use the interactive deployment script:

```bash
# Interactive deployment with guided prompts
python scripts/run_in_vm.py

# Preview commands without executing (dry-run mode)
python scripts/run_in_vm.py --dry-run
```

**Prerequisites:**

- Install required libraries locally:

  ```bash
  pip install rich>=13.7.0 questionary>=2.0.0
  ```

- Have SSH access to an Ubuntu VM with a `.pem` private key file
- VM should have Python 3.8+ installed

**What the script does:**

1. **Collects connection details**: VM ID (ubuntu@host) and PEM file path
2. **Configures environment**: Interactive prompts for WandB logging and AWS S3 upload
3. **Handles checkpoints**: Optionally transfers existing checkpoint for resuming training
4. **Sets up VM**: Clones repository, creates virtual environment, installs dependencies
5. **Starts training**: Runs training in a detached `screen` session so it persists after disconnection

**Key features:**

- Claude Code-style terminal UI with rich panels, tables, and progress indicators
- **Interactive file picker with Tab autocomplete** for PEM files and checkpoints
- Smart environment configuration (only prompts for optional features)
- Automatic dependency installation on VM
- Detached screen session (training continues after closing terminal)
- Dry-run mode to preview all SSH/SCP commands before execution
- Comprehensive error handling with actionable messages

**After deployment:**

Training runs in a screen session named `ddl-{mode}-{version}-{timestamp}`. To reconnect:

```bash
ssh -i <pem_file> ubuntu@host
screen -ls                    # List all screen sessions
screen -r ddl-train-v1-...    # Reconnect to training session
# Press Ctrl+A then D to detach
```

Monitor training progress:

- If WandB is enabled: Check your WandB dashboard for live metrics
- Otherwise: Reconnect to the screen session to view logs

**Example workflow:**

```bash
# First run: fresh training on VM
python scripts/run_in_vm.py
# Answer prompts: VM ID, PEM file, enable WandB, select train mode, etc.

# Later: resume training from checkpoint
python scripts/run_in_vm.py
# When asked about resuming, select Yes and provide checkpoint path
# Script transfers checkpoint and configures VM to resume training
```

---

## Roadmap ideas

- Experiment with deeper stacks / scaling laws under delta gating
- Compare against vanilla GPT residual baseline on TinyStories perplexity
