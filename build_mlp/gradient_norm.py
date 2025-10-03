import numpy as np
import torch.nn as nn
from typing import Any

def log_gradient_norms(
    model: nn.Module,
    step: int,
    logger: Any
) -> None:
    """
    Calculates and logs the gradient norms at iteration=`step` 
    and logs information in `logger`.

    Parameters
    ----------
    model : torch.nn.Module
        The model that will be trained.
    step : int
        The current iteration count.
    logger : Any
        Logger instance that is used to save debugging information.
    """
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = np.linalg.norm(param.grad.data, ord=2)
            total_norm += param_norm**2
            logger.log(f"Epoch: {step + 1}, grad_norm/{name}: {param_norm}")

    logger.log(f"Epoch: {step + 1}, grad_norm/total: {np.sqrt(total_norm)}\n")