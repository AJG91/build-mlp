import torch as tc
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(    
    model: tc.nn.Module,
    loader: tc.utils.data.DataLoader,
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
    device : torch.device or str
        Device on which training is performed.
        Options: "cpu", "cuda"
        
    Returns
    -------
    float
        Validation/test metrics.
    """
    all_preds, all_targets = [], []
    
    model.eval()
    with tc.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            
            all_preds.append(output)
            all_targets.append(y)

    all_preds = tc.cat(all_preds).detach().cpu().numpy().ravel()
    all_targets = tc.cat(all_targets).detach().cpu().numpy().ravel()

    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    return mse, r2


    