import os
import re
import glob
import torch as tc
from typing import Union
from create_checkpoints import CheckpointManager

def load_results(fname, start_dir, headers, *args):
    """
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

def find_file_glob(pattern: str, start_dir: str):
    """
    """
    search = os.path.join(start_dir, "**", pattern)
    found = glob.glob(search, recursive=True)
    return found

def create_table(data: list, headers: list[str], decimals: int = 6) -> None:
    def round_to_decimal(x: Union[int, float]) -> str:
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