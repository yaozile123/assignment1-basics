import math
from typing import Iterable
import torch


def cosine_learning_rate_schedule(current_step: int, max_lr: float, min_lr: float, warmup_steps: int, total_annealing_steps: int) -> float:
    if current_step < warmup_steps:
        return current_step / warmup_steps * max_lr
    elif current_step <= total_annealing_steps:
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / (total_annealing_steps - warmup_steps)))
    else:
        return min_lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total_norm = 0
    parameters_list = list(parameters)
    for p in parameters_list:
        if p.grad is not None:
            total_norm += p.grad.pow(2).sum().item()
    total_norm = total_norm ** 0.5
    if total_norm > max_l2_norm:
        clip_coefficient = max_l2_norm / (total_norm + 1e-6)
        for p in parameters_list:
            if p.grad is not None:
                p.grad.data.mul_(clip_coefficient)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, current_step: int, checkpoint_path: str) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "current_step": current_step,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["current_step"]