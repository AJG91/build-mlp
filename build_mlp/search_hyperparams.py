import os
import torch as tc
from experiment import model_pipeline

def search_hyperparameters(product, cfg, fpath, json_manager, device, plot=False):

    test_metrics = tc.zeros((len(product), 2))

    for i, (vary_prms_i) in enumerate(product):
        print(f"params: {vary_prms_i}")
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