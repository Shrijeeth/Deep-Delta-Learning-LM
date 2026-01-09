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
- `BLOCK_SIZE` (sequence length, default 512)
- `BATCH_SIZE` (default 16)
- `MAX_EPOCHS` (default 3)
- `LR` (default 3e-4)
- `IS_RESUME`, `CHECKPOINT_PATH` for resuming training

---

## Training

```bash
python -m v1.train
```

What happens:

1. Seeds everything, sets matmul precision high.
2. Builds TinyStories datamodule with dynamic padding.
3. Instantiates `DeepLatentGPT` with the delta blocks.
4. Runs Lightning Trainer with checkpointing, LR monitor, optional WandB logger.

To resume:

```bash
IS_RESUME=true CHECKPOINT_PATH=checkpoints_deeplatent/last.ckpt python -m v1.train
```

---

## Inference (generation)

`v1/inference.py` is a stub; typical flow:

```python
import torch
from transformers import AutoTokenizer
from v1.model import DeepLatentGPT, DeepLatentConfig

tokenizer = AutoTokenizer.from_pretrained("gpt2")
config = DeepLatentConfig(vocab_size=tokenizer.vocab_size, block_size=128)
model = DeepLatentGPT(config)
ckpt = torch.load("checkpoints_deeplatent/last.ckpt")
model.load_state_dict(ckpt["state_dict"])
model.eval()

prompt = "Once upon a time"
ids = tokenizer(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    for _ in range(50):
        logits = model(ids)[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)

print(tokenizer.decode(ids[0], skip_special_tokens=True))
```

---

## Project layout

- `v1/model.py` — Deep Delta blocks, GPT backbone, Lightning module
- `v1/train.py` — training script + Trainer setup
- `v1/inference.py` — placeholder for generation helpers
- `data.py` — TinyStories datamodule (tokenize, block, pad)
- `config.py` — environment-driven hyperparameters
- `Makefile` — install/format/lint helpers

---

## References

- Zhang, Yifan; Liu, Yifeng; Wang, Mengdi; Gu, Quanquan. *Deep Delta Learning*. arXiv:2601.00417, 2026.
  [arxiv.org/abs/2601.00417](https://arxiv.org/abs/2601.00417)

---

## Roadmap ideas

- Experiment with deeper stacks / scaling laws under delta gating
- Compare against vanilla GPT residual baseline on TinyStories perplexity
