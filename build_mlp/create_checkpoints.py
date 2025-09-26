import os
import torch as tc
from torch import nn, optim
from typing import Optional, Union, Tuple
from utils import OutputLogger

class CheckpointManager:
    """
    Saves model checkpoints during training.
    Loads different saved checkpoints.

    Saves model checkpoint and optimizer states after each epoch. 
    Keeps only the last `num_keep` checkpoints (rolling window).
    Saves the best model based on lowest validation loss.
    
    Parameters
    ----------
    save_dir : str, optional (default="checkpoints")
        Name of directory where models will be saved.
    num_keep: int, optional (default=5)
        Total number of models to keep.
        Example: Keep last 5 models.
    device : torch.device or str
        Device on which training is performed.
        Options: "cpu", "cuda"
    """
    def __init__(
        self, 
        save_dir: str = "checkpoints", 
        num_keep: int = 5, 
        device: Union[str, tc.device] = "cpu"
    ):
        self.save_dir = save_dir
        self.num_keep = num_keep
        self.device = tc.device(device)
        os.makedirs(save_dir, exist_ok=True)

        self.best_path = os.path.join(save_dir, "metrics_val_best_model.pth")
        self.best_val = None

        if os.path.exists(self.best_path):
            ckpt = tc.load(self.best_path, map_location=self.device)
            self.best_val = float(ckpt.get("val_loss", float("inf")))

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        train_loss: Union[float, tc.Tensor],
        val_loss: Union[float, tc.Tensor],
        logger: OutputLogger,
        fname: str = "last_weights"
    ) -> None:
        """
        Save checkpoint and update best checkpoint if val_loss improves.
    
        Parameters
        ----------
        model : 
        optimizer : 
        epoch : 
        train_loss : 
        val_loss : 
        logger : OutputLogger
        fname : str, optional (default="last_weights")
        """
        if isinstance(val_loss, tc.Tensor):
            val_loss = float(val_loss.detach().cpu().item())
        if isinstance(train_loss, tc.Tensor):
            train_loss = float(train_loss.detach().cpu().item())

        ckpt_path = os.path.join(self.save_dir, f"{fname}_{epoch + 1}.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        tc.save(checkpoint, ckpt_path)

        ckpts = self.sort_files_in_dir(fname)
        if len(ckpts) > self.num_keep:
            os.remove(os.path.join(self.save_dir, ckpts[0]))

        if (self.best_val is None) or (val_loss < self.best_val):
            tc.save(checkpoint, self.best_path)
            self.best_val = val_loss
            logger.log(f"New best model at epoch {epoch + 1} (val_loss={val_loss:.4f})")

    def sort_files_in_dir(self, fname: str) -> list:
        """
        Sorts the files in `self.save_dir` by numbers in file name.
        
        Parameters
        ----------
        fname : str
            Beginning of name of file.

        Returns
        -------
        ckpts : list
            A sorted list of all the files in `self.save_dir` that begin with fname.
        """
        get_number = lambda x: int(x.split("_")[2].split(".")[0])
        ckpts = sorted(
            (f for f in os.listdir(self.save_dir) if f.startswith(f"{fname}_")), key=get_number
        )
        return ckpts
    
    def load_checkpoint(
        self,
        fname: str = "last_weights",
        use_best: bool = False,
    ):
        """
    
        Parameters
        ----------
        fname : str, optional (default="last_weights")
        use_best : boolean, optional (default=False)

        Returns
        -------
        checkpoint : 
        """
        if use_best:
            ckpt_path = self.best_path
            if not os.path.exists(ckpt_path):
                raise ValueError("Model does not exist.")
        else:
            ckpts = self.sort_files_in_dir(fname)
            if not ckpts:
                raise ValueError("Model does not exist.")
            ckpt_path = os.path.join(self.save_dir, ckpts[-1])

        checkpoint = tc.load(ckpt_path, map_location=self.device)
        return checkpoint

    def load_model(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        fname: str = "last_weights",
        use_best: bool = False,
        load_optimizer: bool = True
    ) -> int:
        """
        Loads a checkpoint into model and return the next start epoch.
        Can load either latest model or best model.
    
        Parameters
        ----------
        model : 
        optimizer : 
        fname : str, optional (default="last_weights")
        use_best : 
        load_optimizer : 

        Returns
        -------
        int
            Next epoch to start training from.
        """
        try:
            checkpoint = self.load_checkpoint(fname, use_best)
        except:
            return 1

        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)

        if load_optimizer and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        return checkpoint["epoch"] + 1

    def load_for_training(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        fname: str = "epoch",
        use_best: bool = False,
    ) -> Tuple[nn.Module, optim.Optimizer, int]:
        """
        Loads a checkpoint and return (model, optimizer, start_epoch) in one call.
        Wrapper for load_checkpoint method.

        Parameters
        ----------
        model : nn.Module
            Model to load.
        optimizer : optim.Optimizer
            Optimizer to load state into.
        fname : str, optional (default="epoch")
            Beginning of name of file.
        use_best : bool, optional (default=False)
            If True, loads the best checkpoint.
            If False, loads latest checkpoint.

        Returns
        -------
        Tuple[nn.Module, optim.Optimizer, int]
            Model, optimizer, and next epoch to start training from.
        """
        start_epoch = self.load_checkpoint(model, optimizer, fname=fname, use_best=use_best, load_optimizer=True)
        return model, optimizer, start_epoch