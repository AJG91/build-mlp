import os
import torch as tc
import numpy as np
import random
import yaml
import json
from typing import Any, Dict
from types import SimpleNamespace

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
        return SimpleNamespace(**yaml.safe_load(f))
    
class OutputLogger:
    """
    Logs print outputs.
    If file already exists, the new outputs will
    be appended to the existing file.
    
    Parameters
    ----------
    filename : str
        Name of file.
    """
    def __init__(self, filename: str, show: bool = False):
        self.file = open(filename, "a")
        self.show = show

    def log(self, message: str) -> None:
        """Prints message and writes message to log file."""
        if self.show:
            print(message)
        self.file.write(message + "\n")
        self.file.flush()

class JsonManager(dict):
    """
    Loads, saves, and updates JSON file.
    Inherits from the built-in dict class.

    Everything is done relative to `folder`, which is the global 
    folder where directories will be created when using `update` method.
    
    Attributes
    ----------
    folder : str
        Path to global folder.
    """
    def __init__(self, folder):
        """
        Initialize the JSONFile instance.
        
        Parameters
        ----------
        folder : str
            Path to global folder.
        """
        super().__init__()
        self.folder = folder

    def create_file(self, fpath, fname):
        """
        Creates new directory and JSON file in new directory.

        Parameters
        ----------
        fpath : str
            Path to where the json parameter file will be saved.
            Relative to `self.folder`.
        fname : str
            Name of the json parameter file.
        """
        os.makedirs(fpath, exist_ok=True)
        os.path.join(fpath, fname)

    def save(self, data: dict, fpath: str):
        """
        Saves dictionary as a JSON file. 
        Creates directory if it does not exist.

        Parameters
        ----------
        data : dict
            Dictionary that will be saved as a JSON file.
        fpath : str
            Path to the json parameter file relative to `self.folder'.
        """
        self.create_file(*os.path.split(self.folder + fpath))
        with open(self.folder + fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load(self, fpath: str = "params.json") -> Dict[str, Any]:
        """
        Reads a json file from the given path and returns its contents as a Python dictionary.

        Parameters
        ----------
        fpath : str, optional (default="params.json")
            Path to the json parameter file relative to `self.folder'.

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
            with open(self.folder + fpath, "r") as f:
                self.file = SimpleNamespace(**json.load(f))
                return self.file
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)

    def update(self, fpath, **kwargs):
        """
        Updates the JSON parameter file using `kwargs`.
        Can also create new directories using `fpath` for JSON 
        parameter files.

        Parameters
        ----------
        fpath : str
            Path to the json parameter file relative to `self.folder'.
            Includes file name.
        """
        dict = vars(self.file)
        if kwargs:
            dict = json.loads(json.dumps(dict))
            dict.update(kwargs)
        self.save(dict, fpath)