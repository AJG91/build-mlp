import torch as tc
from experiment import model_pipeline
from typing import Any
from types import SimpleNamespace

def search_hyperparameters(
    product: list, 
    cfg: SimpleNamespace, 
    fpath: str, 
    json_manager: Any,
    device: tc.device, 
    plot: bool = False
) -> tc.Tensor:
    """
    Performs hyperparameter search to identify the optimal 
    configuration for a model.

    This function explores different hyperparameter combinations 
    using grid search and evaluates each configuration on validation 
    and test data to determine the best-performing set of parameters.
    
    Parameters
    ----------
    product : list
        List of different combinations ofparameters that will 
        be used to train the model.
    cfg : SimpleNamespace
        Contains the configuration parameters.
    fpath : str
        Path where metrics and and checkpoints will be stored.
    json_manager : Any
        Instance of the json manager.
        Used to create, update, and save json files.
    device : tc.device
        Device on which training is performed.
        Options: 'cpu', 'cuda'
    plot : bool, optional (default=False)
        If True, plots metrics.
        If False, does not plot metrics.
    
    Returns
    -------
    test_metrics : tc.Tensor
        Outputs a tensor with the testing phase metrics.
    """
    test_metrics = tc.zeros((len(product), 2))

    for i, (vary_prms_i) in enumerate(product):
        print(f"params #{i+1}: {vary_prms_i}")
        lr_i, dim_i, dropout_i = vary_prms_i
        rel_path = (
            f"tests/test{i+1}/"
            f"lr{lr_i}_hdim{dim_i}_do{dropout_i}/"
        )
        fname = f"lr{lr_i}_hdim{dim_i}_do{dropout_i}.json"
        model_vrs = f"test{i+1}_lr{lr_i}_hdim{dim_i}_do{dropout_i}"
        
        json_manager.update(
            fpath + rel_path + fname, 
            model_version=model_vrs,
            learning_rate=lr_i,
            hidden_dim=dim_i,
            dropout=dropout_i
        )
        prms = json_manager.load(fpath + rel_path + fname)

        test_metrics[i] = model_pipeline(prms, cfg, fpath + rel_path, device, plot=plot)

    return test_metrics