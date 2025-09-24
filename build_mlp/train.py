import os
import torch as tc
from sklearn.metrics import r2_score
from typing import Callable

def train_model(
    model: tc.nn.Module,
    loader: tc.utils.data.DataLoader,
    optimizer: tc.optim.Optimizer,
    loss_fn: Callable[[tc.Tensor, tc.Tensor], tc.Tensor],
    device: tc.device,
) -> float:
    """
    Trains a model over one epoch and calculates the accuracy.

    Parameters
    ----------
    model : torch.nn.Module
        The model that will be trained.
    loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    optimizer : torch.optim.Optimizer
        Optimization algorithm for updating model parameters.
    loss_fn : Callable
        Function that will be used to calculate the discrepancy between predictions and targets.
    device : torch.device or str
        Device on which training is performed.
        Options: "cpu", "cuda"
        
    Returns
    -------
    float
        Training metrics.
    """
    train_loss = 0
    all_preds, all_targets = [], []
    
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)

        all_preds.append(output)
        all_targets.append(y)

    all_preds = tc.cat(all_preds).detach().cpu().numpy().ravel()
    all_targets = tc.cat(all_targets).detach().cpu().numpy().ravel()

    train_loss /= len(loader.dataset)
    r2 = r2_score(all_targets, all_preds)
    
    return tc.Tensor([train_loss, r2])
    