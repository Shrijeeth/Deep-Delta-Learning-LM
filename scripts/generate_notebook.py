#!/usr/bin/env python3
"""
Interactive Notebook Generator for Deep-Delta-Learning-LM
Generates environment-specific Jupyter notebooks for Kaggle, Colab, and local Jupyter.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import questionary
from questionary import Style as QuestionaryStyle
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Constants
REPO_URL = "https://github.com/Shrijeeth/Deep-Delta-Learning-LM.git"
REPO_NAME = "Deep-Delta-Learning-LM"


class NotebookGeneratorError(Exception):
    """Custom exception for notebook generation errors."""

    pass


class NotebookGenerator:
    """Generates environment-specific Jupyter notebooks with Claude Code-style UI."""

    def __init__(
        self,
        pre_selected_env: Optional[str] = None,
        pre_selected_version: Optional[str] = None,
    ):
        self.console = Console()
        self.environment: Optional[str] = pre_selected_env
        self.version: Optional[str] = pre_selected_version
        self.features: Dict[str, bool] = {
            "training": True,
            "inference": True,
            "widgets": True,
        }
        self.hyperparams: Dict[str, any] = {}
        self.output_path: Optional[Path] = None
        self.repo_url = REPO_URL

        # Questionary style matching run_in_vm.py theme
        self.questionary_style = QuestionaryStyle(
            [
                ("qmark", "fg:#00d787 bold"),  # Green question mark
                ("question", "bold"),  # Bold question text
                ("answer", "fg:#00d787 bold"),  # Green answer
                ("pointer", "fg:#00d787 bold"),  # Green pointer
                ("highlighted", "fg:#00d787"),  # Green highlight
                ("selected", "fg:#00d787"),  # Green selected
                ("text", ""),  # Plain text
                ("instruction", "fg:#858585"),  # Gray instructions
            ]
        )

    def print_header(self):
        """Display welcome header."""
        self.console.print(
            Panel.fit(
                "🚀 Deep Delta Learning - Jupyter Notebook Generator",
                style="bold magenta",
            )
        )
        self.console.print(
            "\n[dim]Generate optimized notebooks for Kaggle, Colab, or local Jupyter environments[/dim]\n"
        )

    def get_environment_config(self) -> Dict:
        """Returns environment-specific paths and settings."""
        configs = {
            "kaggle": {
                "checkpoint_dir": "/kaggle/working/checkpoints_deeplatent",
                "dataset_cache": "/kaggle/working/datasets",
                "batch_size": 4,
                "num_workers": 0,
                "setup_instructions": [
                    "Enable GPU: Settings → Accelerator → GPU",
                    "Enable Internet: Settings → Internet → On",
                ],
            },
            "colab": {
                "checkpoint_dir": "/content/checkpoints_deeplatent",
                "dataset_cache": "/content/datasets",
                "batch_size": 4,
                "num_workers": 0,
                "setup_instructions": [
                    "Change runtime type: Runtime → Change runtime type → GPU",
                    "Verify GPU with: !nvidia-smi",
                ],
            },
            "jupyter": {
                "checkpoint_dir": "./checkpoints_deeplatent",
                "dataset_cache": "./datasets",
                "batch_size": 16,
                "num_workers": 2,
                "setup_instructions": [
                    "Ensure CUDA is available on your system",
                    "GPU recommended for training",
                ],
            },
        }
        return configs[self.environment]

    def get_environment_defaults(self) -> Dict:
        """Get default hyperparameters based on environment."""
        env_config = self.get_environment_config()
        return {
            "batch_size": env_config["batch_size"],
            "lr": 3e-4,
            "max_epochs": 1 if self.environment in ["kaggle", "colab"] else 3,
            "block_size": 512,
            "data_length": 128,
        }

    def select_environment(self):
        """Step 1: Environment selection."""
        if self.environment:
            return  # Already pre-selected

        self.console.print("\n[bold cyan]Step 1: Select Target Environment[/bold cyan]")

        choices = [
            questionary.Choice(
                "Kaggle - Optimized for Kaggle Notebooks", value="kaggle"
            ),
            questionary.Choice(
                "Google Colab - Optimized for Google Colab", value="colab"
            ),
            questionary.Choice(
                "Local Jupyter - Optimized for local Jupyter", value="jupyter"
            ),
        ]

        self.environment = questionary.select(
            "Choose environment:", choices=choices, style=self.questionary_style
        ).ask()

        if not self.environment:
            raise KeyboardInterrupt

        self.console.print(
            f"[green]✓ Selected: {self.environment.capitalize()}[/green]"
        )

    def select_version(self):
        """Step 1b: Version selection."""
        if self.version:
            return  # Already pre-selected

        self.console.print("\n[bold cyan]Step 1b: Select Model Version[/bold cyan]")

        choices = [
            questionary.Choice("v1 - Original Deep Latent GPT", value="v1"),
            questionary.Choice(
                "v2 - Improved Deep Latent GPT (Recommended)", value="v2"
            ),
        ]

        self.version = questionary.select(
            "Choose version:", choices=choices, style=self.questionary_style
        ).ask()

        if not self.version:
            raise KeyboardInterrupt

        self.console.print(f"[green]✓ Selected: {self.version}[/green]")

    def configure_features(self):
        """Step 2: Feature selection."""
        self.console.print(
            "\n[bold cyan]Step 2: Configure Notebook Features[/bold cyan]"
        )

        features = questionary.checkbox(
            "Select features to include (use Space to select, Enter to confirm):",
            choices=[
                questionary.Choice("Training Pipeline", checked=True, value="training"),
                questionary.Choice(
                    "Inference & Generation", checked=True, value="inference"
                ),
                questionary.Choice(
                    "Interactive Hyperparameter Widgets", checked=True, value="widgets"
                ),
            ],
            style=self.questionary_style,
        ).ask()

        if features is None:
            raise KeyboardInterrupt

        for key in self.features:
            self.features[key] = key in features

        # Ensure at least one feature is selected
        if not any(self.features.values()):
            self.console.print(
                "[yellow]Warning: No features selected, enabling training by default[/yellow]"
            )
            self.features["training"] = True

        self.console.print("[green]✓ Features configured[/green]")

    def configure_hyperparameters(self):
        """Step 3: Hyperparameter configuration."""
        self.console.print("\n[bold cyan]Step 3: Customize Hyperparameters[/bold cyan]")

        # Get defaults based on environment
        defaults = self.get_environment_defaults()

        self.console.print("[dim]Press Enter to use defaults shown in brackets[/dim]\n")

        self.hyperparams = {
            "batch_size": int(
                Prompt.ask("Batch size", default=str(defaults["batch_size"]))
            ),
            "lr": float(Prompt.ask("Learning rate", default=str(defaults["lr"]))),
            "max_epochs": int(
                Prompt.ask("Max epochs", default=str(defaults["max_epochs"]))
            ),
            "block_size": int(
                Prompt.ask(
                    "Block size (context length)", default=str(defaults["block_size"])
                )
            ),
            "data_length": int(
                Prompt.ask(
                    "Data length (tokenization max)",
                    default=str(defaults["data_length"]),
                )
            ),
        }

        self.console.print("[green]✓ Hyperparameters configured[/green]")

    def preview_structure(self):
        """Step 4: Preview notebook structure."""
        self.console.print("\n[bold cyan]Step 4: Notebook Preview[/bold cyan]")

        table = Table(show_header=True, header_style="bold magenta", show_lines=True)
        table.add_column("Section", style="cyan", width=30)
        table.add_column("Description", style="white", width=50)

        table.add_row(
            "1. Title + Instructions",
            f"Project overview for {self.environment.capitalize()} ({self.version})",
        )
        table.add_row(
            "2. Environment Setup",
            f"{self.environment.capitalize()}-specific instructions",
        )
        table.add_row("3. Install Dependencies", "pip install from requirements.txt")
        table.add_row("4. Clone Repository", f"Git clone {self.repo_url}")

        # Platform-specific secrets description
        secrets_desc = {
            "kaggle": "Load secrets from Kaggle Secrets (WANDB, AWS, etc.)",
            "colab": "Load secrets from Colab userdata (WANDB, AWS, etc.)",
            "jupyter": "Load secrets from .env file (WANDB, AWS, etc.)",
        }
        table.add_row("5. Load Secrets", secrets_desc[self.environment])
        table.add_row("6. Environment Variables", "Configure settings and paths")
        table.add_row(
            "7. Import Libraries", f"Import torch, Lightning, {self.version}.model"
        )

        section_num = 8
        if self.features["widgets"]:
            table.add_row(
                f"{section_num}. Hyperparameter Widgets",
                "Interactive sliders for LR, batch size, epochs",
            )
            section_num += 1

        if self.features["training"]:
            table.add_row(
                f"{section_num}. Training Pipeline",
                "Initialize data, model, and run training",
            )
            section_num += 1

        if self.features["inference"]:
            table.add_row(
                f"{section_num}. Inference & Generation",
                "Load checkpoint and generate text",
            )
            section_num += 1

        table.add_row(f"{section_num}. Next Steps", "Links and tips")

        self.console.print(table)

        # Show config summary
        self.console.print("\n[bold cyan]Configuration Summary[/bold cyan]")
        config_table = Table(show_header=True, header_style="bold magenta")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Environment", self.environment.capitalize())
        config_table.add_row("Version", self.version.upper())
        config_table.add_row("Batch Size", str(self.hyperparams["batch_size"]))
        config_table.add_row("Learning Rate", str(self.hyperparams["lr"]))
        config_table.add_row("Max Epochs", str(self.hyperparams["max_epochs"]))
        config_table.add_row("Block Size", str(self.hyperparams["block_size"]))
        config_table.add_row("Data Length", str(self.hyperparams["data_length"]))

        features_enabled = [k.capitalize() for k, v in self.features.items() if v]
        config_table.add_row("Features", ", ".join(features_enabled))

        self.console.print(config_table)

    def generate_markdown_cell(self, content: str) -> Dict:
        """Generate a markdown cell."""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in content.split("\n")],
        }

    def generate_code_cell(self, code: str) -> Dict:
        """Generate a code cell."""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.split("\n")],
        }

    def generate_title_cell(self) -> Dict:
        """Generate title and instructions."""
        env_name = self.environment.capitalize()
        content = f"""# Deep Delta Learning - {env_name} Notebook ({self.version.upper()})

This notebook provides a complete pipeline for training and inference with the Deep Delta Learning language model.

**Model Architecture:** DeepLatentGPT with geometric erase-write updates
**Model Version:** {self.version.upper()}
**Dataset:** TinyStories (via Hugging Face datasets)
**Framework:** PyTorch Lightning

Generated with the interactive notebook generator from [{REPO_NAME}]({self.repo_url})"""
        return self.generate_markdown_cell(content)

    def generate_setup_instructions_cell(self) -> Dict:
        """Generate environment-specific setup instructions."""
        env_config = self.get_environment_config()
        content = (
            "## Environment Setup\n\n**Important:** Before running this notebook:\n\n"
        )
        for instruction in env_config["setup_instructions"]:
            content += f"- {instruction}\n"

        return self.generate_markdown_cell(content)

    def generate_install_dependencies_cell(self) -> Dict:
        """Generate pip install cell."""
        code = """# Install dependencies
!pip install -q lightning==2.6.0 wandb==0.23.1 transformers==4.57.3 datasets==4.4.2 \\
    python-dotenv==1.2.1 pydantic==2.12.5 pydantic-settings==2.12.0 \\
    ipywidgets

print("✓ Dependencies installed successfully!")"""
        return self.generate_code_cell(code)

    def generate_clone_repo_cell(self) -> Dict:
        """Generate repository clone cell."""
        code = f"""# Clone the Deep Delta Learning repository
import os

repo_name = '{REPO_NAME}'
if not os.path.exists(repo_name):
    !git clone {self.repo_url}
    print(f"✓ Repository cloned: {{repo_name}}")
else:
    print(f"✓ Repository already exists: {{repo_name}}")

# Navigate to project directory
os.chdir(repo_name)
print(f"Working directory: {{os.getcwd()}}")"""
        return self.generate_code_cell(code)

    def generate_secrets_loading_cell(self) -> Dict:
        """Generate platform-specific secrets loading cell."""
        if self.environment == "kaggle":
            code = """# Load secrets from Kaggle Secrets
# To add secrets in Kaggle:
# 1. Go to notebook Settings → Add-ons
# 2. Enable "Secrets" and add your keys (e.g., WANDB_API_KEY, AWS_ACCESS_KEY_ID, etc.)

import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

# List of secret keys to load (add more as needed)
SECRET_KEYS = [
    'WANDB_API_KEY',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'HF_TOKEN',  # Hugging Face token
]

print("Loading secrets from Kaggle Secrets:")
for key in SECRET_KEYS:
    try:
        secret_value = user_secrets.get_secret(key)
        if secret_value:
            os.environ[key] = secret_value
            print(f"  ✓ {key} loaded")
    except Exception:
        print(f"  ⚠ {key} not found (skipping)")

print("\\n✓ Secrets loading complete!")"""

        elif self.environment == "colab":
            code = """# Load secrets from Colab Secrets
# To add secrets in Colab:
# 1. Click the key icon (🔑) in the left sidebar
# 2. Add your secrets (e.g., WANDB_API_KEY, AWS_ACCESS_KEY_ID, etc.)
# 3. Enable "Notebook access" toggle for each secret

import os
from google.colab import userdata

# List of secret keys to load (add more as needed)
SECRET_KEYS = [
    'WANDB_API_KEY',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'HF_TOKEN',  # Hugging Face token
]

print("Loading secrets from Colab userdata:")
for key in SECRET_KEYS:
    try:
        secret_value = userdata.get(key)
        if secret_value:
            os.environ[key] = secret_value
            print(f"  ✓ {key} loaded")
    except Exception:
        print(f"  ⚠ {key} not found (skipping)")

print("\\n✓ Secrets loading complete!")"""

        else:  # jupyter (local)
            code = """# Load secrets from .env file
# To use secrets in local Jupyter:
# 1. Create a .env file in the project root
# 2. Add your secrets in KEY=VALUE format:
#    WANDB_API_KEY=your_key_here
#    AWS_ACCESS_KEY_ID=your_key_here
#    AWS_SECRET_ACCESS_KEY=your_secret_here
#    HF_TOKEN=your_token_here

import os
from pathlib import Path

def load_env_file(env_path='.env'):
    \"\"\"Load environment variables from .env file.\"\"\"
    env_file = Path(env_path)
    if not env_file.exists():
        print(f"⚠ No .env file found at {env_path}")
        print("  Create a .env file to store your secrets")
        return

    print(f"Loading secrets from {env_path}:")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                os.environ[key] = value
                print(f"  ✓ {key} loaded")

    print("\\n✓ Secrets loading complete!")

# Load from .env file
load_env_file()"""

        return self.generate_code_cell(code)

    def generate_env_setup_cell(self) -> Dict:
        """Generate environment variables setup cell."""
        env_config = self.get_environment_config()

        code = f"""# Configure environment variables
import os

# Critical: Set NUM_WORKERS={env_config['num_workers']} for {self.environment} environment
os.environ['NUM_WORKERS'] = '{env_config['num_workers']}'

# Checkpoint and cache directories
checkpoint_dir = '{env_config['checkpoint_dir']}'
dataset_cache = '{env_config['dataset_cache']}'

os.environ['CHECKPOINT_DIR'] = checkpoint_dir
os.environ['DATASET_CACHE'] = dataset_cache

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(dataset_cache, exist_ok=True)

# Tokenizer configuration
os.environ['TOKENIZER_NAME'] = 'gpt2'
os.environ['BLOCK_SIZE'] = '{self.hyperparams['block_size']}'
os.environ['DATA_LENGTH'] = '{self.hyperparams['data_length']}'

# Training hyperparameters
# Note: These values will be used by the train() function.
# If you run the widget cell below, widget values will override these.
os.environ['BATCH_SIZE'] = '{self.hyperparams['batch_size']}'
os.environ['LR'] = '{self.hyperparams['lr']}'
os.environ['MAX_EPOCHS'] = '{self.hyperparams['max_epochs']}'
os.environ['MAX_TRAINING_HOURS'] = '5'

# AWS configuration (will be overridden by secrets if loaded)
os.environ.setdefault('AWS_ENABLED', 'false')

# WandB configuration (will be overridden by secrets if loaded)
# Note: WANDB_API_KEY will be loaded from secrets in the previous cell
if 'WANDB_API_KEY' not in os.environ or not os.environ['WANDB_API_KEY']:
    print("⚠ WANDB_API_KEY not set - WandB logging disabled")
    print("  To enable: Add WANDB_API_KEY to your secrets")
else:
    print("✓ WANDB_API_KEY detected - WandB logging enabled")

print(f"\\n✓ Environment configured for {self.environment}!")
print(f"  Checkpoints: {{checkpoint_dir}}")
print(f"  Dataset cache: {{dataset_cache}}")"""
        return self.generate_code_cell(code)

    def generate_imports_cell(self) -> Dict:
        """Generate imports cell."""
        code = f"""# Import required libraries
import sys
import os
import torch

# Add project to path (environment detection)
if 'KAGGLE_CONTAINER_NAME' in os.environ:
    sys.path.insert(0, f'/kaggle/working/{{repo_name}}')
elif 'COLAB_GPU' in os.environ:
    sys.path.insert(0, f'/content/{{repo_name}}')
else:
    sys.path.insert(0, os.getcwd())

# Verify setup
print(f"PyTorch version: {{torch.__version__}}")
print(f"CUDA available: {{torch.cuda.is_available()}}")
print(f"Model version: {self.version}")
if torch.cuda.is_available():
    print(f"CUDA device: {{torch.cuda.get_device_name(0)}}")

print("\\n✓ Environment ready!")"""
        return self.generate_code_cell(code)

    def generate_widget_cell(self) -> Dict:
        """Generate interactive hyperparameter widgets with optional timeout."""
        code = f"""# Interactive Hyperparameter Configuration (Optional)
# This cell provides interactive widgets for tuning hyperparameters.
# If running in background or automated mode, it will auto-proceed after 30 seconds.

import ipywidgets as widgets
from IPython.display import display, HTML
import threading
import time

# Flag to track if user interacted
user_interacted = False
auto_proceed_after = 30  # seconds

# Create sliders
lr_slider = widgets.FloatLogSlider(
    value={self.hyperparams['lr']},
    base=10,
    min=-5,  # 1e-5
    max=-2,  # 1e-2
    step=0.1,
    description='Learning Rate:',
    style={{'description_width': 'initial'}},
    readout_format='.2e'
)

batch_size_slider = widgets.IntSlider(
    value={self.hyperparams['batch_size']},
    min=1,
    max=32,
    step=1,
    description='Batch Size:',
    style={{'description_width': 'initial'}}
)

epochs_slider = widgets.IntSlider(
    value={self.hyperparams['max_epochs']},
    min=1,
    max=10,
    step=1,
    description='Epochs:',
    style={{'description_width': 'initial'}}
)

n_layer_slider = widgets.IntSlider(
    value=8,
    min=4,
    max=16,
    step=2,
    description='Model Layers:',
    style={{'description_width': 'initial'}}
)

n_embd_slider = widgets.IntSlider(
    value=384,
    min=128,
    max=768,
    step=128,
    description='Embedding Dim:',
    style={{'description_width': 'initial'}}
)

# Status display
status_label = widgets.HTML(
    value=f"<p style='color: #00d787;'>⏱ Adjust sliders within {{auto_proceed_after}} seconds, or they'll auto-apply with default values</p>"
)

# Apply widget values to environment
def apply_hyperparams(change=None):
    global user_interacted
    user_interacted = True
    os.environ['LR'] = str(lr_slider.value)
    os.environ['BATCH_SIZE'] = str(batch_size_slider.value)
    os.environ['MAX_EPOCHS'] = str(epochs_slider.value)
    print(f"✓ Applied: LR={{lr_slider.value:.2e}}, Batch={{batch_size_slider.value}}, Epochs={{epochs_slider.value}}")

# Auto-proceed timer
def auto_proceed_timer():
    for remaining in range(auto_proceed_after, 0, -1):
        if user_interacted:
            status_label.value = "<p style='color: #00d787;'>✓ User interaction detected - values will be applied</p>"
            return
        status_label.value = f"<p style='color: #ffa500;'>⏱ Auto-proceeding in {{remaining}} seconds (adjust sliders to customize)</p>"
        time.sleep(1)

    if not user_interacted:
        apply_hyperparams()
        status_label.value = "<p style='color: #00d787;'>✓ Auto-applied default values</p>"

# Auto-apply on widget change
lr_slider.observe(apply_hyperparams, names='value')
batch_size_slider.observe(apply_hyperparams, names='value')
epochs_slider.observe(apply_hyperparams, names='value')

# Display widgets
display(status_label)
display(lr_slider, batch_size_slider, epochs_slider, n_layer_slider, n_embd_slider)

# Start auto-proceed timer in background thread
timer_thread = threading.Thread(target=auto_proceed_timer, daemon=True)
timer_thread.start()

print(f"Hyperparameter widgets ready. Default values will apply in {{auto_proceed_after}}s if unchanged.")"""
        return self.generate_code_cell(code)

    def generate_training_cells(self) -> List[Dict]:
        """Generate training pipeline cells."""
        cells = []

        # Section header
        cells.append(
            self.generate_markdown_cell(
                """## Training Pipeline

This section runs the training pipeline using the project's train module."""
            )
        )

        # Training code - simplified
        train_code = f"""# Run training
from {self.version}.train import train

print("\\n" + "="*60)
print("Starting training...")
print("="*60 + "\\n")

# Train using the official training function
# It will read configuration from environment variables set above
train()

print("\\n" + "="*60)
print("✓ Training complete!")
print("="*60)"""

        cells.append(self.generate_code_cell(train_code))

        return cells

    def generate_inference_cells(self) -> List[Dict]:
        """Generate inference and generation cells."""
        cells = []

        # Section header
        cells.append(
            self.generate_markdown_cell(
                """## Inference & Text Generation

Load a trained checkpoint and generate text."""
            )
        )

        env_config = self.get_environment_config()

        # Set checkpoint path
        setup_code = f"""# Configure inference
import os

# Set checkpoint path
checkpoint_path = '{env_config['checkpoint_dir']}/last.ckpt'
os.environ['CHECKPOINT_PATH'] = checkpoint_path

if not os.path.exists(checkpoint_path):
    print(f"⚠ Checkpoint not found at {{checkpoint_path}}")
    print("Available checkpoints:")
    !ls -lh {env_config['checkpoint_dir']}
else:
    print(f"✓ Checkpoint found: {{checkpoint_path}}")"""

        cells.append(self.generate_code_cell(setup_code))

        # Run inference with default prompt
        inference_code = f"""# Run inference with default prompt
from {self.version}.inference import run_inference

print("\\n" + "="*60)
print("Running inference...")
print("="*60 + "\\n")

run_inference()"""

        cells.append(self.generate_code_cell(inference_code))

        # Interactive custom prompt
        interactive_code = f"""# Custom prompt inference
# Note: You can modify the prompt below and run this cell multiple times

custom_prompt = "In a magical forest,"  # Change this prompt as desired

print(f"\\nGenerating with custom prompt: '{{custom_prompt}}'\\n")
print("─"*60)

# Run inference with custom prompt
from {self.version}.inference import run_inference
run_inference(prompt=custom_prompt)

print("─"*60)"""

        cells.append(self.generate_code_cell(interactive_code))

        return cells

    def build_notebook_json(self) -> Dict:
        """Build complete notebook JSON structure."""
        cells = []

        # 1. Title and instructions
        cells.append(self.generate_title_cell())

        # 2. Setup instructions
        cells.append(self.generate_setup_instructions_cell())

        # 3. Install dependencies
        cells.append(self.generate_markdown_cell("## Installation"))
        cells.append(self.generate_install_dependencies_cell())

        # 4. Clone repository
        cells.append(self.generate_markdown_cell("## Clone Repository"))
        cells.append(self.generate_clone_repo_cell())

        # 5. Load secrets
        cells.append(self.generate_markdown_cell("## Load Secrets"))
        cells.append(self.generate_secrets_loading_cell())

        # 6. Environment setup
        cells.append(self.generate_markdown_cell("## Environment Configuration"))
        cells.append(self.generate_env_setup_cell())

        # 7. Imports
        cells.append(self.generate_markdown_cell("## Import Libraries"))
        cells.append(self.generate_imports_cell())

        # 8. Widgets (if enabled)
        if self.features["widgets"]:
            cells.append(self.generate_markdown_cell("## Hyperparameter Configuration"))
            cells.append(self.generate_widget_cell())

        # 9. Training (if enabled)
        if self.features["training"]:
            cells.extend(self.generate_training_cells())

        # 10. Inference (if enabled)
        if self.features["inference"]:
            cells.extend(self.generate_inference_cells())

        # 11. Next steps
        next_steps_content = f"""## Next Steps

- Experiment with different hyperparameters using the {"widgets above" if self.features['widgets'] else "environment variables"}
- Try different prompts for text generation
- Monitor training progress (checkpoints saved in `{self.get_environment_config()['checkpoint_dir']}`)
- Read more about Deep Delta Learning: [arXiv:2601.00417](https://arxiv.org/abs/2601.00417)

**Repository:** {self.repo_url}"""
        cells.append(self.generate_markdown_cell(next_steps_content))

        # Build final notebook structure
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.10.0",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        return notebook

    def write_notebook(self, notebook_json: Dict):
        """Write notebook JSON to .ipynb file."""
        # Generate filename
        filename = f"deep_delta_learning_{self.version}_{self.environment}.ipynb"

        # Determine output directory (current directory)
        output_dir = Path.cwd()
        self.output_path = output_dir / filename

        # Write notebook
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(notebook_json, f, indent=2, ensure_ascii=False)

        self.console.print(f"\n[green]✓ Notebook written to {self.output_path}[/green]")

    def show_success_message(self):
        """Display success message and next steps."""
        env_instructions = {
            "kaggle": "1. Upload to Kaggle Notebooks\n2. Enable GPU and Internet in Settings\n3. Add secrets (WANDB_API_KEY, etc.) in Settings → Secrets\n4. Run cells sequentially",
            "colab": "1. Upload to Google Colab\n2. Change runtime type to GPU\n3. Add secrets via 🔑 icon (WANDB_API_KEY, etc.)\n4. Run cells sequentially",
            "jupyter": f"1. Create a .env file with your secrets\n2. Open with: jupyter notebook {self.output_path.name}\n3. Ensure CUDA is available\n4. Run cells sequentially",
        }

        secrets_info = {
            "kaggle": "Add secrets in Kaggle Settings → Secrets",
            "colab": "Add secrets via 🔑 icon in left sidebar",
            "jupyter": "Create .env file with KEY=VALUE format",
        }

        instructions = f"""[bold cyan]Notebook Generated Successfully![/bold cyan]

[bold]Output:[/bold] {self.output_path}
[bold]Environment:[/bold] {self.environment.capitalize()}
[bold]Version:[/bold] {self.version.upper()}
[bold]Size:[/bold] {self.output_path.stat().st_size / 1024:.1f} KB

[bold cyan]Next Steps:[/bold cyan]

{env_instructions[self.environment]}

[bold cyan]Tips:[/bold cyan]
• Model version: {self.version.upper()}
• NUM_WORKERS is set to {self.get_environment_config()['num_workers']} (optimized for {self.environment})
• Checkpoints saved to {self.get_environment_config()['checkpoint_dir']}
• Secrets loading included: {secrets_info[self.environment]}
• {"Interactive widgets enabled for hyperparameter tuning" if self.features['widgets'] else "Configure hyperparameters in environment setup"}
• {"Training pipeline included - ready to train!" if self.features['training'] else "Training pipeline not included"}
• {"Inference pipeline included - load checkpoint and generate!" if self.features['inference'] else "Inference pipeline not included"}
"""

        self.console.print(
            Panel(instructions, title="[green]Success[/green]", style="green")
        )

    def run(self):
        """Execute the full notebook generation workflow."""
        try:
            self.print_header()
            self.select_environment()
            self.select_version()
            self.configure_features()
            self.configure_hyperparameters()
            self.preview_structure()

            # Final confirmation
            if not Confirm.ask(
                "\n[bold yellow]Generate notebook?[/bold yellow]", default=True
            ):
                self.console.print("[yellow]Generation cancelled[/yellow]")
                return

            with self.console.status(
                "[cyan]Building notebook...[/cyan]", spinner="dots"
            ):
                notebook_json = self.build_notebook_json()

            self.write_notebook(notebook_json)
            self.show_success_message()

            self.console.print(
                "\n[bold green]✓ Generation completed successfully![/bold green]"
            )

        except NotebookGeneratorError as e:
            self.console.print(f"\n[bold red]✗ Generation failed: {e}[/bold red]")
            sys.exit(1)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Generation interrupted by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"\n[bold red]✗ Unexpected error: {e}[/bold red]")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate environment-specific Jupyter notebooks for Deep Delta Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python scripts/generate_notebook.py

  # Generate directly for specific environment
  python scripts/generate_notebook.py --env colab

  # Generate for specific environment and version
  python scripts/generate_notebook.py --env colab --version v2

Environment options:
  - kaggle: Optimized for Kaggle Notebooks
  - colab: Optimized for Google Colab
  - jupyter: Optimized for local Jupyter

Version options:
  - v1: Original Deep Latent GPT
  - v2: Improved Deep Latent GPT (Recommended)
        """,
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["kaggle", "colab", "jupyter"],
        help="Target environment (skip interactive selection)",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v2"],
        help="Model version (skip interactive selection)",
    )

    args = parser.parse_args()

    generator = NotebookGenerator(
        pre_selected_env=args.env, pre_selected_version=args.version
    )

    # Show pre-selected options
    if args.env:
        generator.console.print(
            f"[cyan]Pre-selected environment: {args.env.capitalize()}[/cyan]"
        )
    if args.version:
        generator.console.print(
            f"[cyan]Pre-selected version: {args.version.upper()}[/cyan]"
        )

    generator.run()


if __name__ == "__main__":
    main()
