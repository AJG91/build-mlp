
import torch as tc
import torch.nn as nn
import torch.optim as optim
from model import MLP
from train import train_model
from evaluate import evaluate_model

def model_pipeline(train, val, test, prms, device):
    X, _ = next(iter(train))
    model, optimizer, loss_fn = initialize_model(X.shape[1], prms, device)
    training(model, train, val, optimizer, loss_fn, prms, device)
    evaluation(model, test, device)
    return None

def initialize_model(input_dim, prms, device):
    model = MLP(input_dim, prms.hidden_dim, prms.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=prms.learning_rate)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn

def training(model, train, val, optimizer, loss_fn, prms, device):
    for e in range(prms.epochs):
        train_mse, train_r2 = train_model(model, train, optimizer, loss_fn, device)
        val_mse, val_r2 = evaluate_model(model, val, device)
        
        print(f"Epoch {e + 1}:")
        print(f"Train - MSE: {train_mse:.4f}, R2-score: {train_r2:.4f}")
        print(f"Validation - MSE: {val_mse:.4f}, R2-score: {val_r2:.4f}\n")
    return None

def evaluation(model, test, device):
    test_mse, test_r2 = evaluate_model(model, test, device)
    print(f"Test - MSE: {test_mse:.4f}, R2-score: {test_r2:.4f}")
    return None

def log_results():
    return None
    