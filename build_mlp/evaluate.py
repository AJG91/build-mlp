import torch as tc
from typing import Callable
from sklearn.metrics import r2_score

def evaluate_model(    
    model: tc.nn.Module,
    loader: tc.utils.data.DataLoader,
    loss_fn: Callable[[tc.Tensor, tc.Tensor], tc.Tensor],
    device: tc.device,
) -> float:
    """
    Evaluates a model and calculates the metrics.
    Can be used for evaluating both the validation and test set.

    Parameters
    ----------
    model : torch.nn.Module
        The model that will be trained.
    loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    loss_fn : Callable
        Function that will be used to calculate the discrepancy between predictions and targets.
    device : torch.device or str
        Device on which training is performed.
        Options: "cpu", "cuda"
        
    Returns
    -------
    float
        Validation/test metrics.
    """
    eval_loss = 0
    all_preds, all_targets = [], []
    
    model.eval()
    with tc.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)

            eval_loss += loss.item() * X.size(0)
            
            all_preds.append(output)
            all_targets.append(y)

    all_preds = tc.cat(all_preds).detach().cpu().numpy().ravel()
    all_targets = tc.cat(all_targets).detach().cpu().numpy().ravel()

    eval_loss /= len(loader.dataset)
    r2 = r2_score(all_targets, all_preds)
    
    return tc.Tensor([eval_loss, r2])


    