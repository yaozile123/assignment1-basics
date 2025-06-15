import torch
import torch.nn as nn
from collections.abc import Iterable
from typing import Callable


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between logits and targets.
    o_i' = o_i - max(o_i)
    loss = -o_i' + log(sum(exp(o_i')))
    
    Args:
        logits (torch.Tensor): The logits of the model.
        targets (torch.Tensor): The targets of the model.
    """
    logits_stable = logits - logits.max(dim=-1, keepdim=True).values # (B, D)
    logits_exp_sum = logits_stable.exp().sum(dim=-1, keepdim=True) # (B, 1)
    loss = -logits_stable[torch.arange(logits.size(0)), targets] + torch.log(logits_exp_sum)
    return loss.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float):
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Callable[[], float]) -> None:
        loss = None if not closure else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                p.data -= lr * p.grad / (t + 1) ** 0.5
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float, betas: tuple[float, float], eps: float = 1e-8, weight_decay: float = 0.01):        
        defaults = {"lr": lr, "eps": eps, "weight_decay": weight_decay}
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        # AdamW algorithm:
        # 1. m ← β1*m + (1-β1)*g     (Update first moment estimate)
        # 2. v ← β2*v + (1-β2)*g²    (Update second moment estimate)
        # 3. α_t ← α * √(1-β2^t) / (1-β1^t)  (Compute bias-corrected learning rate)
        # 4. θ ← θ - α_t * m / (√v + ε)      (Update parameters)
        # 5. θ ← θ - α*λ*θ                   (Apply weight decay)

        """
        loss = None if not closure else closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                m = self.beta1 * m + (1 - self.beta1) * p.grad
                v = self.beta2 * v + (1 - self.beta2) * p.grad**2
                lr_t = lr * (1 - self.beta2**t)**0.5 / (1 - self.beta1**t)
                p.data -= lr_t * m / (v.sqrt() + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss