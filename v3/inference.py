import os
from typing import Optional

import torch
from transformers import AutoTokenizer

from config import get_settings
from v3.model import DeepLatentConfig, DeepLatentGPT


def run_inference(
    prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    debug: Optional[bool] = None,
):
    """
    Lightweight inference helper for DeepLatentGPT v2.

    Args:
        prompt: Text prompt for generation (default: "Once upon a " or INFERENCE_PROMPT env var)
        max_new_tokens: Number of tokens to generate (default: 50)
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k sampling parameter (default: 50)
        repetition_penalty: Repetition penalty (default: 1.1)
        debug: Enable debug output (default: True)
    """
    settings = get_settings()

    # --- User-adjustable defaults ---
    DEFAULT_PROMPT = "Once upon a"
    DEFAULT_MAX_NEW_TOKENS = 50
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_TOP_K = 50
    DEFAULT_REPETITION_PENALTY = 1.2
    DEFAULT_DEBUG = False
    # --------------------------------

    # Use provided parameters or check environment, fallback to defaults
    PROMPT = (
        prompt
        if prompt is not None
        else os.environ.get("INFERENCE_PROMPT", DEFAULT_PROMPT)
    )
    CHECKPOINT_PATH = settings.CHECKPOINT_PATH
    MAX_NEW_TOKENS = (
        max_new_tokens if max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    )
    TEMPERATURE = temperature if temperature is not None else DEFAULT_TEMPERATURE
    TOP_K = top_k if top_k is not None else DEFAULT_TOP_K
    REPETITION_PENALTY = (
        repetition_penalty
        if repetition_penalty is not None
        else DEFAULT_REPETITION_PENALTY
    )
    DEBUG = debug if debug is not None else DEFAULT_DEBUG

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
        debug=DEBUG,
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\nPrompt: {PROMPT}\n---\n{output_text}\n")
    return output_text
