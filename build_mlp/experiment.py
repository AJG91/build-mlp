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
from plots import plot_metrics_vs_epochs

def model_pipeline(prms, cfg, out_path, device, show=False, plot=False):
    logger = OutputLogger(out_path + "/output.log", show=show)

    train, val, test = get_dataloaders(cfg.dataset, cfg.normalize, 
                                       cfg.data_split, cfg.batch_size, 
                                       cfg.seed)

    X, _ = next(iter(train))
    model, optimizer, loss_fn = initialize_model(X.shape[1], prms, device)
    train_metrics, val_metrics = training(model, train, val, optimizer, 
                                          loss_fn, prms.epochs, logger, 
                                          out_path + "/checkpoints", device)
    test_metrics = evaluation(model, test, loss_fn, logger, device)

    if plot:
        rel_path = cfg.fig_path + out_path.split('/', 2)[-1]
        os.makedirs(rel_path, exist_ok=True)

        plot_metrics_vs_epochs(train_metrics[:, 0], val_metrics[:, 0], 
                               prms.epochs, rel_path, cfg.dpi, "Huber")
        plot_metrics_vs_epochs(train_metrics[:, 1], val_metrics[:, 1], 
                               prms.epochs, rel_path, cfg.dpi, "R2")
    return test_metrics

def initialize_model(input_dim, prms, device):
    model = MLP(input_dim, prms.hidden_dim, prms.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=prms.learning_rate)
    loss_fn = nn.SmoothL1Loss(beta=1.0) 
    return model, optimizer, loss_fn

def training(model, train, val, optimizer, loss_fn, epochs, logger, path, device):
    manager = CheckpointManager(path)
    train_metrics = tc.zeros((epochs, 2))
    val_metrics = tc.zeros((epochs, 2))
    
    for e in range(epochs):
        logger.log(f"Epoch {e + 1}:")

        train_metrics[e] = train_model(model, train, optimizer, loss_fn, device)
        val_metrics[e] = evaluate_model(model, val, loss_fn, device)
        manager.save(model, optimizer, e, train_metrics[e][0], val_metrics[e][0], logger)

        logger.log(f"Train - Huber: {train_metrics[e][0]:.4f}, "
                   f"R2-score: {train_metrics[e][1]:.4f}")
        logger.log(f"Validation - Huber: {val_metrics[e][0]:.4f}, "
                   f"R2-score: {val_metrics[e][1]:.4f}\n")
    return train_metrics, val_metrics

def evaluation(model, test, loss_fn, logger, device):
    test_metrics = evaluate_model(model, test, loss_fn, device)
    logger.log(f"Test - Huber: {test_metrics[0]:.4f}, "
               f"R2-score: {test_metrics[1]:.4f}\n")
    return test_metrics
    