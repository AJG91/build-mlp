import os
import torch as tc
import torch.nn as nn
import torch.optim as optim
from model import MLP
from train import train_model
from evaluate import evaluate_model
from build_dataset import get_dataloaders
from create_checkpoints import CheckpointManager
from utils import OutputLogger
from gradient_norm import log_gradient_norms
from plots import plot_metrics_vs_epochs
from typing import Any, Callable, Tuple
from types import SimpleNamespace

def model_pipeline(
    prms: SimpleNamespace, 
    cfg: SimpleNamespace, 
    out_path: str, 
    device: tc.device,
    show: bool = False, 
    plot: bool = False
) -> tc.Tensor:
    """
    Executes the complete machine learning pipeline, 
    including producing the dataloaders, model training and 
    evaluation, and plot generation.
    
    Parameters
    ----------
    prms : SimpleNamespace
        Contains the model parameters.
    cfg : SimpleNamespace
        Contains the configuration parameters.
    out_path : str
        Path where metrics and and checkpoints will be stored.
    device : tc.device
        Device on which training is performed.
        Options: 'cpu', 'cuda'
    show : bool, optional (default= False)
        Prints out results to the notebook.
    plot : bool, optional (default= False)
        If True, plots metrics.
        If False, does not plot metrics.
    
    Returns
    -------
    test_metrics : tc.Tensor
        Outputs a tensor with the testing phase metrics.
    """
    metrics_logger = OutputLogger(out_path + "/metrics_output.log", show=show)
    grad_logger = OutputLogger(out_path + "/grad_norm_output.log", show=show)

    train, val, test = get_dataloaders(cfg.dataset, cfg.normalize, 
                                       cfg.data_split, cfg.batch_size, 
                                       cfg.seed)

    X, _ = next(iter(train))
    model, optimizer, loss_fn = initialize_model(X.shape[1], prms, device)

    train_metrics, val_metrics = training(model, train, val, optimizer, 
                                          loss_fn, prms.epochs, 
                                          metrics_logger, grad_logger, 
                                          out_path + "/checkpoints", device)
    test_metrics = evaluation(model, test, loss_fn, metrics_logger, device)

    if plot:
        rel_path = cfg.fig_path + out_path.split('/', 2)[-1]
        os.makedirs(rel_path, exist_ok=True)

        plot_metrics_vs_epochs(train_metrics[:, 0], val_metrics[:, 0], 
                               prms.epochs, rel_path, cfg.dpi, "Huber")
        plot_metrics_vs_epochs(train_metrics[:, 1], val_metrics[:, 1], 
                               prms.epochs, rel_path, cfg.dpi, "R2")
    return test_metrics

def initialize_model(
    input_dim: int, 
    prms: SimpleNamespace, 
    device: tc.device,
) -> Tuple[nn.Module, optim.Optimizer, Callable]:
    """
    Initializes the MLP model, optimizer, and loss function.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input data.
    prms : SimpleNamespace
        Contains the model parameters.
    device : torch.device or str
        Device on which training is performed.
        Options: 'cpu', 'cuda'
    
    Returns
    -------
    model : nn.Module
    optimizer : optim.Optimizer
    loss_fn : Callable
    """
    model = MLP(input_dim, prms.hidden_dim, prms.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=prms.learning_rate)
    loss_fn = nn.SmoothL1Loss(beta=1.0) 
    return model, optimizer, loss_fn

def training(
    model: tc.nn.Module,
    train: tc.utils.data.DataLoader, 
    val: tc.utils.data.DataLoader, 
    optimizer: tc.optim.Optimizer,
    loss_fn: Callable[[tc.Tensor, tc.Tensor], tc.Tensor],
    epochs: int, 
    metrics_logger: Any, 
    debug_logger: Any, 
    path: str, 
    device: tc.device,
) -> Tuple[tc.Tensor, tc.Tensor]:
    """
    Performs the training and validation phase of the machine 
    learning pipeline, including logging the metrics.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model that will be trained.
    train : tc.utils.data.DataLoader
        DataLoader for the training set, with shuffling enabled.
    val : tc.utils.data.DataLoader
        DataLoader for the validation set, without shuffling.
    optimizer : tc.optim.Optimizer
        Optimizer used by the model.
    loss_fn : Callable
        Function that will be used to calculate the discrepancy between predictions and targets.
    metrics_logger : Any
        Logger instance that is used to save training information.
    debug_logger : Any
        Logger instance that is used to save debugging information.
    path : str
        Path to directory where checkpoints will be saved.
    device : torch.device or str
        Device on which training is performed.
        Options: 'cpu', 'cuda'

    Returns
    -------
    train_metrics : tc.Tensor
        Outputs a tensor with the training phase metrics.
    val_metrics : tc.Tensor
        Outputs a tensor with the validation phase metrics.
    """
    manager = CheckpointManager(path)
    train_metrics = tc.zeros((epochs, 2))
    val_metrics = tc.zeros((epochs, 2))
    
    for e in range(epochs):
        metrics_logger.log(f"Epoch {e + 1}:")

        train_metrics[e] = train_model(model, train, optimizer, loss_fn, device)
        val_metrics[e] = evaluate_model(model, val, loss_fn, device)
        manager.save(model, optimizer, e, train_metrics[e][0], val_metrics[e][0], metrics_logger)
        log_gradient_norms(model, e, debug_logger)

        metrics_logger.log(f"Train - Huber: {train_metrics[e][0]:.4f}, "
                           f"R2-score: {train_metrics[e][1]:.4f}")
        metrics_logger.log(f"Validation - Huber: {val_metrics[e][0]:.4f}, "
                           f"R2-score: {val_metrics[e][1]:.4f}\n")
    return train_metrics, val_metrics

def evaluation(
    model: tc.nn.Module, 
    eval, 
    loss_fn: Callable[[tc.Tensor, tc.Tensor], tc.Tensor],
    logger: Any, 
    device: tc.device
) -> tc.Tensor:
    """
    Evaluates the performance of the model with unseen data.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model that will be trained.
    eval : tc.utils.data.DataLoader
        DataLoader for the evaluation phase.
    loss_fn : Callable
        Function that will be used to calculate the discrepancy between predictions and targets.
    logger : Any
        Logger instance that is used to save evaluation information.
    device : torch.device or str
        Device on which training is performed.
        Options: 'cpu', 'cuda'
    
    Returns
    -------
    eval_metrics : tc.Tensor
        Outputs a tensor with the evaluation phase metrics.
    """
    eval_metrics = evaluate_model(model, eval, loss_fn, device)
    logger.log(f"Test - Huber: {eval_metrics[0]:.4f}, "
               f"R2-score: {eval_metrics[1]:.4f}\n")
    return eval_metrics