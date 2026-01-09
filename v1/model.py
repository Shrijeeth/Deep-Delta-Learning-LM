import math
from dataclasses import dataclass

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.text.perplexity import Perplexity


# --- Configuration ---
@dataclass
class DeepLatentConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 8  # Deep Latent models often benefit from more depth
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False  # Modern architectures often disable bias in Linear
    beta_init: float = -2.0  # Initialize gate to be nearly closed (Identity)
    lr: float = 3e-4


# --- The "Delta" Latent Update Mechanism ---
class DeltaLatentUpdate(nn.Module):
    """
    The heart of DeepLatentGPT.
    Implements the 'Erase-Write' geometric update rule.
    """

    def __init__(self, config):
        super().__init__()
        # 1. The 'Key' direction k: What feature are we editing?
        # A small bottleneck MLP helps find the precise direction in the latent space.
        self.k_proj = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4, bias=False),
            nn.SiLU(),  # Swish activation
            nn.Linear(config.n_embd // 4, config.n_embd, bias=False),
        )

        # 2. The 'Gate' beta: How strongly do we edit?
        # Scalar per token.
        self.beta_proj = nn.Linear(config.n_embd, 1, bias=True)

        # 3. The 'Value' v: What is the new target magnitude?
        self.v_proj = nn.Linear(config.n_embd, 1, bias=False)

        # Init beta to negative so sigmoid(beta) starts near 0 (Conservative updates)
        nn.init.constant_(self.beta_proj.bias, config.beta_init)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, x, residual_signal):
        # x: The current Latent State [Batch, Seq, Dim]
        # residual_signal: The proposed update from Attention/MLP [Batch, Seq, Dim]

        # A. Determine Direction (k)
        k = self.k_proj(x)
        k = F.normalize(k, p=2, dim=-1)  # Must be unit vector for geometric projection

        # B. Determine Gate Strength (beta)
        # 2 * sigmoid allows range [0, 2].
        # 0 = Keep state. 1 = Erase & Replace. 2 = Reflect/Invert.
        beta = 2.0 * torch.sigmoid(self.beta_proj(x))

        # C. Determine Target Value (v)
        # The Attention/MLP output provides the "intensity" of the new feature.
        v = self.v_proj(residual_signal)

        # D. Geometric Update
        # 1. How much of 'k' is currently in 'x'? (Projection)
        x_on_k = (x * k).sum(dim=-1, keepdim=True)

        # 2. The Delta: Scale by beta, point along k, move from current(x_on_k) to target(v)
        delta = beta * k * (v - x_on_k)

        return x + delta


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(
                torch.tril(torch.ones(T, T, device=x.device)) == 0, float("-inf")
            )
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 4x expansion standard for GPT
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class DeepLatentBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.delta_attn = DeltaLatentUpdate(config)  # The Latent Editor

        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
        self.delta_mlp = DeltaLatentUpdate(config)  # The Latent Editor

    def forward(self, x):
        # 1. Self-Attention Sub-Block
        # Standard: x = x + attn(ln(x))
        # DeepLatent: x = DeltaUpdate(x, attn(ln(x)))
        attn_signal = self.attn(self.ln1(x))
        x = self.delta_attn(x, attn_signal)

        # 2. Feed-Forward Sub-Block
        # Standard: x = x + mlp(ln(x))
        # DeepLatent: x = DeltaUpdate(x, mlp(ln(x)))
        mlp_signal = self.mlp(self.ln2(x))
        x = self.delta_mlp(x, mlp_signal)
        return x


class DeepLatentGPT(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [DeepLatentBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=LayerNorm(config.n_embd, config.bias),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Tie weights

        self.perplexity_metric = Perplexity(
            ignore_index=-100,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()

        # Positional Encoding
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        # Initialize Latent State
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through Deep Latent Blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def training_step(self, batch, batch_idx):
        idx, targets = batch["input_ids"], batch["labels"]
        logits = self(idx)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
        )
        perplexity = self.perplexity_metric(logits, targets)
        self.log("train/loss", loss)
        self.log("train/perplexity", perplexity)
        return loss

    def validation_step(self, batch, batch_idx):
        idx, targets = batch["input_ids"], batch["labels"]
        logits = self(idx)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
        )
        perplexity = self.perplexity_metric(logits, targets)
        self.log("val/loss", loss)
        self.log("val/perplexity", perplexity)
        return loss

    def configure_optimizers(self):
        # Weight decay setup
        param_dict = {pn: p for pn, p in self.named_parameters()}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 1e-1},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return AdamW(optim_groups, lr=self.config.lr, betas=(0.9, 0.95))
