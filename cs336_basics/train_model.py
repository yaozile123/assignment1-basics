import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import HfArgumentParser

from cs336_basics.data import data_loading
from cs336_basics.layers import TransformerLM
from cs336_basics.optimizer import AdamW, cross_entropy_loss
from cs336_basics.train_utils import (
    cosine_learning_rate_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# parsing the training configuration
@dataclass
class TrainingConfig:
    # dataset parameters
    train_data_path: str
    val_data_path: str
    context_length: int = field(default=256)
    batch_size: int = field(default=32)
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")

    # checkpoint parameters
    checkpoint_dir: str = field(default="checkpoints")
    resume_from_checkpoint: str = field(default=None)

    # model parameters (default values from GPT2 config)
    vocab_size: int = field(default=32000)
    d_model: int = field(default=512)
    num_layers: int = field(default=6)
    num_heads: int = field(default=8)
    d_ff: int = field(default=2048)

    # training parameters
    max_steps: int = field(default=100000)
    learning_rate: float = field(default=1e-3)
    min_learning_rate: float = field(default=0)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    weight_decay: float = field(default=0.01)
    grad_clip: float = field(default=1.0)
    warmup_steps: int = field(default=None)
    theta: int = field(default=10000)  # for rope

    # logging parameters
    log_interval: int = field(default=None)
    eval_interval: int = field(default=None)
    eval_iters: int = field(default=None)
    no_wandb: bool = field(default=False)
    wandb_project: str = field(default="cs336-basics")
    wandb_run_name: str = field(default=None)

    def __post_init__(self):
        # Validate arguments
        if self.warmup_steps is None:
            self.warmup_steps = int(self.max_steps * 0.01)
        if self.log_interval is None:
            self.log_interval = int(self.max_steps * 0.01)
        if self.eval_interval is None:
            self.eval_interval = int(self.max_steps * 0.1)
        if self.eval_iters is None:
            self.eval_iters = 50  # Set a reasonable default value


def eval_model(model, config, val_data, device, step, checkpoint_dir, optimizer):
    model.eval()
    val_loss = 0.0
    for _ in range(config.eval_iters):
        inputs, targets = data_loading(
            dataset=val_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )
        logits = model(inputs)
        loss = cross_entropy_loss(logits.view(-1, config.vocab_size), targets.view(-1))
        val_loss += loss.item()
    val_loss /= config.eval_iters
    logging.info(f"Validation Loss: {val_loss:.4f}")
    if not config.no_wandb:
        wandb.log({"val/loss": val_loss}, step=step + 1)
    checkpoint_path = checkpoint_dir / f"checkpoint_{step + 1}_lr_{config.learning_rate}.pth"
    logging.info(f"Saving checkpoint to {checkpoint_path}")
    save_checkpoint(model, optimizer, step + 1, checkpoint_path)
    model.train()
    return val_loss


def train(config: TrainingConfig):
    """
    Main training loop.
    """
    # Initialize wandb
    if not config.no_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config),
        )

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device(config.device)
    logging.info(f"Using device: {device}")

    # Create DataLoaders
    logging.info("Loading datasets...")
    train_data = np.load(config.train_data_path, mmap_mode="r")
    val_data = np.load(config.val_data_path, mmap_mode="r")
    logging.info(
        f"Datasets loaded. Train: {len(train_data)} tokens, Val: {len(val_data)} tokens"
    )

    # Model initialization
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        theta=config.theta,
    ).to(device)
    if device == "mps":
        model = torch.compile(model, backend="aot_eager")
    logging.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters."
    )

    # Optimizer initialization
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )

    # Load checkpoint if exists
    start_step = 0
    checkpoint_path = None
    if config.resume_from_checkpoint:
        if Path(config.resume_from_checkpoint).is_file():
            checkpoint_path = config.resume_from_checkpoint
        else:
            logging.warning(
                f"Specified checkpoint not found: {config.resume_from_checkpoint}. "
                "Attempting to load latest from checkpoint directory."
            )

    if checkpoint_path:
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        start_step = load_checkpoint(checkpoint_path, model, optimizer)

    model.train()
    running_loss = 0.0

    for step in tqdm(range(start_step, config.max_steps)):
        # Get a batch of data
        inputs, targets = data_loading(
            dataset=train_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )

        # Forward pass
        logits = model(inputs)
        loss = cross_entropy_loss(logits.view(-1, config.vocab_size), targets.view(-1))
        running_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            gradient_clipping(model.parameters(), config.grad_clip)
        optimizer.step()

        lr = cosine_learning_rate_schedule(
            current_step=step,
            max_lr=config.learning_rate,
            min_lr=config.min_learning_rate,
            warmup_steps=config.warmup_steps,
            total_annealing_steps=config.max_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Logging
        if (step + 1) % config.log_interval == 0:
            avg_loss = running_loss / config.log_interval
            logging.info(
                f"Step {step + 1}/{config.max_steps} | LR: {lr:.6f} | Train Loss: {avg_loss:.4f}"
            )
            if not config.no_wandb:
                wandb.log(
                    {"train/loss": avg_loss, "train/lr": lr},
                    step=step + 1,
                )
            running_loss = 0.0

        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            eval_model(model, config, val_data, device, step, checkpoint_dir, optimizer)


if __name__ == "__main__":
    # parsing config
    parser = HfArgumentParser(TrainingConfig)
    config = parser.parse_args_into_dataclasses()[0]
    logging.info(f"Training with config: {asdict(config)}")

    train(config)
