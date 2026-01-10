#!/usr/bin/env python3
"""
Interactive VM Deployment Script for Deep-Delta-Learning-LM
Deploys and runs training on a remote Ubuntu VM with a Claude Code-style terminal UI.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import questionary
from questionary import Style as QuestionaryStyle
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Constants
REPO_URL = "https://github.com/Shrijeeth/Deep-Delta-Learning-LM.git"
REPO_NAME = "Deep-Delta-Learning-LM"
PROJECT_DIR = f"~/{REPO_NAME}"
CHECKPOINT_DIR = "checkpoints_deeplatent"


class VMDeploymentError(Exception):
    """Custom exception for VM deployment errors."""

    pass


class VMDeployer:
    """Handles VM deployment operations with rich terminal UI."""

    def __init__(self, dry_run: bool = False):
        self.console = Console()
        self.dry_run = dry_run
        self.vm_id: Optional[str] = None
        self.pem_file: Optional[str] = None
        self.env_config: dict = {}
        self.is_resume: bool = False
        self.checkpoint_path: Optional[str] = None
        self.remote_checkpoint_path: Optional[str] = None
        self.mode: Optional[str] = None
        self.version: Optional[str] = None

        # Questionary style matching rich theme
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
            Panel.fit("🚀 Deep Delta Learning - VM Deployment", style="bold magenta")
        )
        if self.dry_run:
            self.console.print(
                Panel("[yellow]DRY RUN MODE - No commands will be executed[/yellow]"),
                style="yellow",
            )

    def validate_vm_id(self, vm_id: str) -> bool:
        """Validate VM ID format (user@host)."""
        if "@" not in vm_id:
            return False
        user, host = vm_id.split("@", 1)
        return bool(user and host)

    def validate_pem_file(self, pem_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate PEM file exists and has correct permissions.
        Returns (is_valid, warning_message).
        """
        if not os.path.exists(pem_path):
            return False, "PEM file not found: {}".format(pem_path)

        if not os.path.isfile(pem_path):
            return False, "Path is not a file: {}".format(pem_path)

        # Check permissions (should be 400)
        stat_info = os.stat(pem_path)
        perms = oct(stat_info.st_mode)[-3:]
        if perms != "400":
            return (
                True,
                f"Warning: PEM file permissions are {perms}, should be 400. Run: chmod 400 {pem_path}",
            )

        return True, None

    def validate_checkpoint_file(self, ckpt_path: str) -> bool:
        """Validate checkpoint file exists and is a .ckpt file."""
        if not os.path.exists(ckpt_path):
            return False
        if not ckpt_path.endswith(".ckpt"):
            return False
        return os.path.isfile(ckpt_path)

    def pick_file(
        self,
        message: str,
        default_path: str = "~",
        file_extension: Optional[str] = None,
        allow_manual: bool = True,
    ) -> str:
        """
        Interactive file picker with tab autocomplete and manual entry fallback.

        Args:
            message: Prompt message to display
            default_path: Starting directory for browsing
            file_extension: Optional file extension filter (e.g., ".pem", ".ckpt")
            allow_manual: Allow user to type 'manual' to enter path manually

        Returns:
            str: Selected file path (expanded and validated)
        """
        self.console.print(f"\n[bold cyan]{message}[/bold cyan]")

        if allow_manual:
            self.console.print(
                "[dim]💡 Hint: Use Tab for autocomplete, or type 'manual' to enter path manually[/dim]"
            )
        else:
            self.console.print("[dim]💡 Hint: Use Tab for autocomplete[/dim]")

        while True:
            # Use questionary for path selection with autocomplete
            result = questionary.path(
                "",  # Empty prompt (message shown above)
                default=default_path,
                only_files=True,
                style=self.questionary_style,
            ).ask()

            # Handle Ctrl+C cancellation
            if result is None:
                raise KeyboardInterrupt

            result = result.strip()

            # Check for manual entry request
            if allow_manual and result.lower() == "manual":
                result = Prompt.ask("Enter file path")

            # Expand ~ to home directory
            result = os.path.expanduser(result)

            # Validate file exists
            if not os.path.exists(result):
                self.console.print(f"[red]✗ File not found: {result}[/red]")
                if not Confirm.ask("Try again?", default=True):
                    raise VMDeploymentError(f"File not found: {result}")
                continue

            # Validate it's a file (not directory)
            if not os.path.isfile(result):
                self.console.print(f"[red]✗ Path is not a file: {result}[/red]")
                continue

            # Validate file extension if specified
            if file_extension and not result.endswith(file_extension):
                self.console.print(
                    f"[yellow]⚠ Warning: File doesn't have {file_extension} extension[/yellow]"
                )
                if not Confirm.ask("Continue anyway?", default=False):
                    continue

            return result

    def collect_vm_connection(self):
        """Collect and validate VM connection details."""
        self.console.print("\n[bold cyan]Step 1: VM Connection Setup[/bold cyan]")

        # Get VM ID (still text input - no file picker needed)
        while True:
            vm_id = Prompt.ask("Enter VM ID (format: ubuntu@host)")
            if self.validate_vm_id(vm_id):
                self.vm_id = vm_id
                break
            self.console.print("[red]✗ Invalid VM ID format. Must be user@host[/red]")

        # Get PEM file with interactive picker
        while True:
            pem_file = self.pick_file(
                message="Select PEM file",
                default_path="~/.ssh",  # Common location for SSH keys
                file_extension=".pem",
            )

            is_valid, message = self.validate_pem_file(pem_file)

            if not is_valid:
                self.console.print(f"[red]✗ {message}[/red]")
                continue

            self.pem_file = pem_file

            if message:  # Warning about permissions
                self.console.print(f"[yellow]{message}[/yellow]")

            break

        self.console.print("[green]✓ VM connection details validated[/green]")

    def load_env_sample(self) -> dict:
        """Load .env.sample as base configuration."""
        env_sample_path = Path(__file__).parent.parent / ".env.sample"

        if not env_sample_path.exists():
            raise VMDeploymentError(f".env.sample not found at {env_sample_path}")

        config = {}
        with open(env_sample_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()

        return config

    def build_env_config(self):
        """Build environment configuration interactively."""
        self.console.print("\n[bold cyan]Step 2: Environment Configuration[/bold cyan]")

        # Load base config from .env.sample
        try:
            self.env_config = self.load_env_sample()
        except VMDeploymentError as e:
            self.console.print(f"[red]✗ {e}[/red]")
            sys.exit(1)

        # WandB configuration
        if Confirm.ask("Enable WandB logging?", default=True):
            wandb_key = Prompt.ask(
                "Enter WandB API key (or press Enter to skip)", default=""
            )
            self.env_config["WANDB_API_KEY"] = wandb_key
        else:
            self.env_config["WANDB_API_KEY"] = ""

        # AWS S3 configuration
        if Confirm.ask(
            "Enable AWS S3 checkpoint upload after training?", default=False
        ):
            self.env_config["AWS_ENABLED"] = "true"
            self.env_config["AWS_ACCESS_KEY_ID"] = Prompt.ask("AWS Access Key ID")
            self.env_config["AWS_SECRET_ACCESS_KEY"] = Prompt.ask(
                "AWS Secret Access Key", password=True
            )
            self.env_config["AWS_BUCKET_NAME"] = Prompt.ask(
                "S3 Bucket Name", default="deep-delta-learning-lm"
            )
            endpoint = Prompt.ask(
                "S3 Endpoint URL (optional, press Enter to skip)", default=""
            )
            self.env_config["AWS_ENDPOINT_URL"] = endpoint
        else:
            self.env_config["AWS_ENABLED"] = "false"

        self.console.print("[green]✓ Environment configuration built[/green]")

    def ask_resume_training(self):
        """Ask if user wants to resume training and collect checkpoint if yes."""
        self.console.print("\n[bold cyan]Step 3: Training Configuration[/bold cyan]")

        self.is_resume = Confirm.ask("Resume from existing checkpoint?", default=False)

        if self.is_resume:
            # Suggest common checkpoint locations
            project_root = Path(__file__).parent.parent
            default_ckpt_dir = project_root / "checkpoints_deeplatent"
            if not default_ckpt_dir.exists():
                default_ckpt_dir = project_root / "v1" / "checkpoints"
            if not default_ckpt_dir.exists():
                default_ckpt_dir = Path.home()

            while True:
                ckpt_path = self.pick_file(
                    message="Select checkpoint file",
                    default_path=str(default_ckpt_dir),
                    file_extension=".ckpt",
                )

                if self.validate_checkpoint_file(ckpt_path):
                    self.checkpoint_path = ckpt_path
                    # Get file size
                    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                    self.console.print(
                        f"[green]✓ Checkpoint validated ({size_mb:.1f} MB)[/green]"
                    )
                    break

                self.console.print(
                    "[red]✗ Invalid checkpoint file. Must exist and end with .ckpt[/red]"
                )

    def ssh_command(self, command: str, description: str = "") -> Tuple[int, str, str]:
        """
        Execute SSH command on VM.
        Returns (return_code, stdout, stderr).
        """
        ssh_cmd = [
            "ssh",
            "-i",
            self.pem_file,
            "-o",
            "StrictHostKeyChecking=no",
            self.vm_id,
            command,
        ]

        if self.dry_run:
            self.console.print(
                Panel(
                    f"[yellow]Would execute:\n  {' '.join(ssh_cmd)}[/yellow]",
                    title="[yellow]DRY RUN[/yellow]",
                )
            )
            return 0, "", ""

        if description:
            with self.console.status(f"[cyan]{description}...[/cyan]", spinner="dots"):
                result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        else:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True)

        return result.returncode, result.stdout, result.stderr

    def scp_transfer(self, local_path: str, remote_path: str, description: str = ""):
        """Transfer file via SCP."""
        scp_cmd = [
            "scp",
            "-i",
            self.pem_file,
            "-o",
            "StrictHostKeyChecking=no",
            local_path,
            f"{self.vm_id}:{remote_path}",
        ]

        if self.dry_run:
            self.console.print(
                Panel(
                    f"[yellow]Would execute:\n  {' '.join(scp_cmd)}[/yellow]",
                    title="[yellow]DRY RUN[/yellow]",
                )
            )
            return 0

        if description:
            with self.console.status(f"[cyan]{description}...[/cyan]", spinner="dots"):
                result = subprocess.run(scp_cmd)
        else:
            result = subprocess.run(scp_cmd)

        return result.returncode

    def transfer_checkpoint(self):
        """Transfer checkpoint to VM."""
        if not self.is_resume or not self.checkpoint_path:
            return

        self.console.print("\n[bold cyan]Transferring checkpoint to VM[/bold cyan]")

        # Create remote checkpoint directory
        remote_dir = f"{PROJECT_DIR}/{CHECKPOINT_DIR}"
        returncode, _, stderr = self.ssh_command(
            f"mkdir -p {remote_dir}", "Creating checkpoint directory on VM"
        )

        if returncode != 0 and not self.dry_run:
            raise VMDeploymentError(f"Failed to create directory: {stderr}")

        # Get checkpoint filename
        ckpt_filename = os.path.basename(self.checkpoint_path)
        remote_ckpt_path = f"{remote_dir}/{ckpt_filename}"

        # Transfer checkpoint
        size_mb = os.path.getsize(self.checkpoint_path) / (1024 * 1024)
        returncode = self.scp_transfer(
            self.checkpoint_path,
            remote_ckpt_path,
            f"Transferring checkpoint ({size_mb:.1f} MB)",
        )

        if returncode != 0 and not self.dry_run:
            raise VMDeploymentError("Failed to transfer checkpoint")

        # Store relative path for .env
        self.remote_checkpoint_path = f"{CHECKPOINT_DIR}/{ckpt_filename}"
        self.console.print(
            f"[green]✓ Checkpoint transferred to {remote_ckpt_path}[/green]"
        )

    def setup_vm_repository(self):
        """Clone or update repository on VM."""
        self.console.print(
            "\n[bold cyan]Step 4: Setting up repository on VM[/bold cyan]"
        )

        # Check if repo exists
        returncode, stdout, _ = self.ssh_command(
            f"test -d {PROJECT_DIR} && echo exists", "Checking for existing repository"
        )

        repo_exists = "exists" in stdout

        if repo_exists:
            self.console.print(
                "[yellow]Repository already exists, pulling latest changes...[/yellow]"
            )
            returncode, _, stderr = self.ssh_command(
                f"cd {PROJECT_DIR} && git pull origin main", "Pulling latest changes"
            )
            if returncode != 0 and not self.dry_run:
                self.console.print(
                    f"[yellow]Warning: git pull failed: {stderr}[/yellow]"
                )
        else:
            self.console.print("Cloning repository...")
            returncode, _, stderr = self.ssh_command(
                f"git clone {REPO_URL}", "Cloning repository"
            )
            if returncode != 0 and not self.dry_run:
                raise VMDeploymentError(f"Failed to clone repository: {stderr}")

        self.console.print("[green]✓ Repository ready[/green]")

    def install_dependencies(self):
        """Install dependencies on VM."""
        self.console.print("\n[bold cyan]Installing dependencies on VM[/bold cyan]")

        # Check if venv exists
        returncode, stdout, _ = self.ssh_command(
            f"test -d {PROJECT_DIR}/venv && echo exists"
        )

        venv_exists = "exists" in stdout

        if not venv_exists:
            self.console.print("Creating virtual environment...")
            returncode, _, stderr = self.ssh_command(
                f"cd {PROJECT_DIR} && python3 -m venv venv",
                "Creating virtual environment",
            )
            if returncode != 0 and not self.dry_run:
                raise VMDeploymentError(f"Failed to create venv: {stderr}")

        # Install dependencies
        returncode, _, stderr = self.ssh_command(
            f"cd {PROJECT_DIR} && source venv/bin/activate && pip install -r requirements.txt",
            "Installing Python dependencies (this may take a few minutes)",
        )
        if returncode != 0 and not self.dry_run:
            self.console.print(
                f"[yellow]Warning: pip install had issues: {stderr}[/yellow]"
            )

        # Install screen if not present
        returncode, stdout, _ = self.ssh_command("which screen")
        if returncode != 0 and not self.dry_run:
            self.console.print("Installing screen utility...")
            self.ssh_command(
                "sudo apt-get update && sudo apt-get install -y screen",
                "Installing screen",
            )

        self.console.print("[green]✓ Dependencies installed[/green]")

    def transfer_env_file(self):
        """Transfer and configure .env file on VM."""
        self.console.print(
            "\n[bold cyan]Step 5: Configuring environment on VM[/bold cyan]"
        )

        # Update env config with IS_RESUME and CHECKPOINT_PATH
        self.env_config["IS_RESUME"] = str(self.is_resume).lower()
        if self.is_resume and self.remote_checkpoint_path:
            self.env_config["CHECKPOINT_PATH"] = self.remote_checkpoint_path
        else:
            self.env_config["CHECKPOINT_PATH"] = ""

        # Write to temp file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_env_path = f"/tmp/ddl_env_{timestamp}"

        with open(temp_env_path, "w") as f:
            for key, value in self.env_config.items():
                f.write(f"{key}={value}\n")

        # Transfer to VM
        remote_env_path = f"{PROJECT_DIR}/.env"
        returncode = self.scp_transfer(
            temp_env_path, remote_env_path, "Transferring environment configuration"
        )

        # Clean up temp file
        if os.path.exists(temp_env_path):
            os.remove(temp_env_path)

        if returncode != 0 and not self.dry_run:
            raise VMDeploymentError("Failed to transfer .env file")

        self.console.print("[green]✓ Environment configured[/green]")

    def ask_execution_params(self):
        """Ask for execution mode and version."""
        self.console.print("\n[bold cyan]Step 6: Execution Parameters[/bold cyan]")

        # Mode selection
        mode = Prompt.ask(
            "Select mode", choices=["train", "inference"], default="train"
        )
        self.mode = mode

        # Version selection (currently only v1, but structured for future)
        version = Prompt.ask("Select version", choices=["v1"], default="v1")
        self.version = version

        self.console.print("[green]✓ Execution parameters set[/green]")

    def show_configuration_summary(self):
        """Display configuration summary table."""
        self.console.print("\n[bold cyan]Configuration Summary[/bold cyan]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("VM", self.vm_id)
        table.add_row("Mode", self.mode)
        table.add_row("Version", self.version)
        table.add_row("Resume", "Yes" if self.is_resume else "No")

        if self.is_resume:
            table.add_row("Checkpoint", self.remote_checkpoint_path)

        table.add_row(
            "WandB Enabled", "Yes" if self.env_config.get("WANDB_API_KEY") else "No"
        )
        table.add_row("AWS S3 Enabled", self.env_config.get("AWS_ENABLED", "false"))

        self.console.print(table)

    def start_training_in_screen(self):
        """Start training in a detached screen session."""
        self.console.print("\n[bold cyan]Step 7: Starting training[/bold cyan]")

        # Generate unique session name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_name = f"ddl-{self.mode}-{self.version}-{timestamp}"

        # Build training command
        train_cmd = f"cd {PROJECT_DIR} && source venv/bin/activate && python main.py --version {self.version} --mode {self.mode}"

        # Create screen session with training command
        screen_cmd = f'screen -dmS {session_name} bash -c "{train_cmd}"'

        returncode, _, stderr = self.ssh_command(
            screen_cmd, f"Starting {self.mode} in screen session"
        )

        if returncode != 0 and not self.dry_run:
            raise VMDeploymentError(f"Failed to start screen session: {stderr}")

        # Show success message and reconnect instructions
        self.console.print(
            f"\n[green]✓ Training started in screen session: {session_name}[/green]"
        )

        # Display reconnect instructions
        instructions = f"""
[bold cyan]To reconnect to the training session:[/bold cyan]
  ssh -i {self.pem_file} {self.vm_id}
  screen -r {session_name}

[bold cyan]To view all screen sessions:[/bold cyan]
  screen -ls

[bold cyan]To detach from screen:[/bold cyan]
  Press Ctrl+A then D

[bold cyan]Monitor training progress:[/bold cyan]"""

        if self.env_config.get("WANDB_API_KEY"):
            instructions += "\n  Check WandB dashboard: https://wandb.ai/<your-project>"
        else:
            instructions += "\n  Reconnect to screen session to view logs"

        self.console.print(
            Panel(instructions, title="[green]Next Steps[/green]", style="green")
        )

    def run(self):
        """Execute the full deployment workflow."""
        try:
            self.print_header()
            self.collect_vm_connection()
            self.build_env_config()
            self.ask_resume_training()

            if self.is_resume:
                self.transfer_checkpoint()

            self.setup_vm_repository()
            self.install_dependencies()
            self.transfer_env_file()
            self.ask_execution_params()
            self.show_configuration_summary()

            # Final confirmation
            if not self.dry_run:
                if not Confirm.ask(
                    "\n[bold yellow]Proceed with deployment?[/bold yellow]",
                    default=True,
                ):
                    self.console.print("[yellow]Deployment cancelled[/yellow]")
                    return

            self.start_training_in_screen()

            self.console.print(
                "\n[bold green]✓ Deployment completed successfully![/bold green]"
            )

        except VMDeploymentError as e:
            self.console.print(f"\n[bold red]✗ Deployment failed: {e}[/bold red]")
            sys.exit(1)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Deployment interrupted by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"\n[bold red]✗ Unexpected error: {e}[/bold red]")
            if not self.dry_run:
                raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy Deep-Delta-Learning-LM to a remote VM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal deployment
  python scripts/run_in_vm.py

  # Preview commands without executing
  python scripts/run_in_vm.py --dry-run
        """,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview commands without executing them"
    )

    args = parser.parse_args()

    deployer = VMDeployer(dry_run=args.dry_run)
    deployer.run()


if __name__ == "__main__":
    main()
