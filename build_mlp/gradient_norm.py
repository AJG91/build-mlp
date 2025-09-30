import numpy as np
import torch.nn as nn
from typing import Any

def log_gradient_norms(
    model: nn.Module,
    step: int,
    logger: Any
) -> None:
    """
    """
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = np.linalg.norm(param.grad.data, ord=2)
            total_norm += param_norm**2
            logger.log(f"Epoch: {step + 1}, grad_norm/{name}: {param_norm}")

    logger.log(f"Epoch: {step + 1}, grad_norm/total: {np.sqrt(total_norm)}\n")