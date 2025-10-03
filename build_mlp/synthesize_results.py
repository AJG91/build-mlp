import os
import re
import glob
import torch as tc
from typing import Union
from create_checkpoints import CheckpointManager

def load_results(
    fname: str, 
    start_dir: str, 
    headers: list[str], 
    *args: tc.Tensor
) -> None:
    """
    Loads the models and results from the hyperparameter exploration.

    Parameters
    ----------
    fname : str
        Name of directory that contains the models.
    start_dir : str
        Initial directory where the different experiments are contained.
    headers : list[str]
        Headers that will be used for displaying a table with results
        for the different hyperparameter experiments.
    *args : tc.Tensor
        Additional metrics that will be displayed.
    """
    models, results, idcs = [], [], []
    files = find_file_glob(fname, start_dir)
    
    for file in files:
        manager = CheckpointManager(file)
        checkpoints = manager.load_checkpoint(use_best=True)
        rel_folders = file.split(start_dir)[-1].split("/")

        if "base_model" in rel_folders:
            models.append(rel_folders[0].replace("'", ""))
            idcs.append(None)
        else:
            models.append(rel_folders[2].replace("'", ""))
            idcs.append(int(re.findall(r"\d+", rel_folders[1])[0]) - 1)

        results.append(checkpoints["val_loss"])

    data = [models] + [results]

    if args:
        base_res = tc.stack(args)[:, 0].numpy().tolist()
        res_args = list(arg[1:][idcs[1:]].numpy().tolist() for arg in args)
        base_res_args = [[a] + b for a, b in zip(base_res, res_args)]
        data += base_res_args

    create_table([list(row) for row in zip(*data)], headers)

def find_file_glob(
    fname: str, 
    start_dir: str
) -> list[str]:
    """
    Stores the different paths leading to the checkpoints
    for all the hyperparameter exploration experiments.
    
    Parameters
    ----------
    fname : str
        Name of directory that contains the models.
    start_dir : str
        Initial directory where the different experiments are contained.

    Returns
    -------
    found : list[str]
        List of the different paths leading to the checkpoints
        for all the hyperparameter experiments.
    """
    search = os.path.join(start_dir, "**", fname)
    found = glob.glob(search, recursive=True)
    return found

def create_table(
    data: list, 
    headers: list[str], 
    decimals: int = 6
) -> None:
    """
    Creates a table with the metrics for all hyperparameter
    experiments. 
    Table compatible with markdown.
    
    Parameters
    ----------
    data : list
        List containing the name of each experiment.
        Name is the file name where the experiment info is contained.
    headers : list[str]
        Headers that will be used for the table.
    decimals : int, optional (default=6)
        Specifies the number of decimals to display.
    """
    def round_to_decimal(x: Union[int, float]) -> str:
        """
        Rounds a number to a fixed number of decimal places 
        and returns it as a string.
        """
        if isinstance(x, (int, float)):
            return f"{x:.{decimals}f}"
        return str(x)

    data = [[round_to_decimal(item) for item in row] for row in data]

    cols = list(zip(*data))
    col_widths = [max(len(str(item)) for item in col) for col in cols]
    col_widths = [max(len(h), w) for h, w in zip(headers, col_widths)]

    header_row = "| " + " | ".join(f"{h:{"^"}{w}}" 
                                   for h, w in zip(headers, col_widths)) + " |"

    separator_row = "| " + " | ".join("-" * w for w in col_widths) + " |"

    data_rows = [
        "| " + " | ".join(f"{str(item):{"^"}{w}}" 
                          for item, w in zip(row, col_widths)) + " |" 
                          for row in data
    ]

    print(header_row)
    print(separator_row)
    for row in data_rows:
        print(row)