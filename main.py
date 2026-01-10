import argparse
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


def main(
    version: str = "v1",
    mode: Literal["train", "inference"] = "train",
):
    if version == "v1":
        if mode == "train":
            from v1.train import train

            train()
        elif mode == "inference":
            from v1.inference import run_inference

            run_inference()
    elif version == "v2":
        if mode == "train":
            from v2.train import train

            train()
        elif mode == "inference":
            from v2.inference import run_inference

            run_inference()
    else:
        raise ValueError(f"Unknown version: {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="train: run training; inference: generate text",
    )
    args = parser.parse_args()

    main(
        version=args.version,
        mode=args.mode,
    )
