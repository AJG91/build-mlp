import torch.nn as nn
from typing import Any

def log_gradient_norms(
    model: nn.Module,
    step: int,
    logger: Any
) -> None:
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            logger.log(f"Epoch: {step + 1}, grad_norm/{name}: {param_norm.item()}")

    logger.log(f"Epoch: {step + 1}, grad_norm/total: {total_norm ** 0.5}\n")