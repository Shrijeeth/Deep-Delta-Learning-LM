import os

import torch
from transformers import AutoTokenizer

from config import get_settings
from v2.model import DeepLatentConfig, DeepLatentGPT


def run_inference():
    """
    Lightweight inference helper for DeepLatentGPT v1.
    Uses defaults defined here; adjust PROMPT / CHECKPOINT_PATH / generation params below.
    """
    settings = get_settings()

    # --- User-adjustable defaults ---
    PROMPT = "Once upon a "
    CHECKPOINT_PATH = settings.CHECKPOINT_PATH
    MAX_NEW_TOKENS = 50
    TEMPERATURE = 1.0
    TOP_K = 50  # e.g., 50 for top-k sampling
    REPETITION_PENALTY = 1.0
    # --------------------------------

    tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = DeepLatentConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=settings.BLOCK_SIZE,
    )
    model = DeepLatentGPT(config)

    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found. Set CHECKPOINT_PATH in .env or edit inference.py (got: {CHECKPOINT_PATH})."
        )

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        debug=True,
    )
    print(output_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\nPrompt: {PROMPT}\n---\n{output_text}\n")
    return output_text
