import torch as tc
import numpy as np
import random
import yaml
import json
from typing import Any, Dict

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed across libraries to ensure reproducibility.

    Initializes the random number generators for:
    - PyTorch (CPU and CUDA, if available)
    - NumPy
    - Python's built-in `random` module

    Parameters
    ----------
    seed : int, optional (default=42)
        The seed value to set for all random number generators.
    """
    tc.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)

def load_json(path: str = "params.json") -> Dict[str, Any]:
    """
    Reads a json file from the given path and returns its contents as a Python dictionary. 

    Parameters
    ----------
    path : str, optional (default="params.json")
        Path to the json parameter file.

    Returns
    -------
    dict
        A dictionary containing the configuration parameters parsed
        from the json file.

    Raises
    ------
    json.JSONDecodeError
        If the specified file does not exist.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print("Invalid JSON:", e)

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Reads a YAML file from the given path and returns its contents as a Python dictionary. 

    Parameters
    ----------
    path : str, optional (default="config.yaml")
        Path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing the configuration parameters parsed
        from the YAML file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    yaml.YAMLError
        If the file cannot be parsed as valid YAML.
    """
    with open(path) as f:
        return yaml.safe_load(f)
    