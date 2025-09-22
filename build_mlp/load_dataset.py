import numpy as np
import torch as tc
from typing import Tuple

class LoadDataset():
    """
    Load and return dataset as PyTorch tensors.

    Attributes
    ----------
    dataset : str
        Name of the dataset to load (e.g., "HOUSING", "MNIST", "MEDMNIST").
    X : tc.Tensor
        Input features of the dataset as a PyTorch tensor.
        shape: (num_samples, channels, height, width)
    y : tc.Tensor
        Target labels as a PyTorch tensor.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load.
        Options: "HOUSING", MNIST", or "MEDMNIST"
    **kwargs
        Additional keyword arguments passed to the specific dataset loader methods.

    Raises
    ------
    ValueError
        If an unsupported dataset name is provided.
    """
    def __init__(self, dataset: str, **kwargs):
        self.dataset = dataset.upper()
        self.loaders = {
            "HOUSING": lambda: self.load_california_housing(**kwargs),
            "MNIST": lambda: self.load_mnist(**kwargs),
            "MEDMNIST": lambda: self.load_medmnist(**kwargs),
        }

        if self.dataset not in self.loaders:
            raise ValueError(f"Dataset not supported -> {dataset}. "
                             f"Available: {list(self.loaders.keys())}")

        self.X, self.y = self.loaders[self.dataset]()

    def load_california_housing(self,) -> Tuple[tc.Tensor, tc.Tensor]:
        """
        Load the California housing dataset and convert it to PyTorch tensors.
            
        Returns
        -------
        X : tc.Tensor
            Features as a float32 tensor.
            Shape: (events, features)
        y : tc.Tensor
            Targets as a float32 tensor.
            Shape: (events, 1)
        """
        from sklearn.datasets import fetch_california_housing
        
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        
        X = tc.tensor(X, dtype=tc.float32)
        y = tc.from_numpy(y.astype("float32"))
        return X, y.unsqueeze(1)
        
    def load_mnist(self, version: int = 1) -> Tuple[tc.Tensor, tc.Tensor]:
        """
        Load the MNIST dataset and convert it to PyTorch tensors.

        Parameters
        ----------
        version : int, optional (default=1)
            The version of the MNIST dataset from OpenML.

        Returns
        -------
        X : tc.Tensor
            MNIST images as a float32 tensor normalized to [0, 1].
            Shape: (num_samples, 1, 28, 28)
        y : tc.Tensor
            MNIST labels as a int64 tensor.
            Shape: (num_samples,)
        """
        from sklearn.datasets import fetch_openml
        
        X, y = fetch_openml("mnist_784", version=version, return_X_y=True, as_frame=False)
        X = tc.tensor(X, dtype=tc.float32)
        y = tc.from_numpy(y.astype("int64"))
        return X.view(-1, 1, 28, 28), y
        
    def load_medmnist(self, flag: str = "bloodmnist") -> Tuple[tc.Tensor, tc.Tensor]:
        """
        Load a MedMNIST dataset and convert it to PyTorch tensors.

        Parameters
        ----------
        flag : str, optional (default="bloodmnist")
            Specifies which MedMNIST dataset to load.
            Must be one of the keys in `medmnist.INFO`.

        Returns
        -------
        X : tc.Tensor
            MedMNIST images as a float32 tensor normalized to [0, 1].
            Shape: (num_samples, channels, height, width)
        y : tc.Tensor
            MedMNIST labels as long tensors.
            Shape: (num_samples,)

        Raises
        ------
        ValueError
            If the specified `flag` does not correspond to a valid MedMNIST dataset.
        """
        import medmnist
        from medmnist import INFO
        
        if flag not in INFO:
            raise ValueError(f"Invalid MedMNIST flag -> {flag}. "
                             f"Available datasets: {list(INFO.keys())}")

        print(INFO[flag]["label"])
        DataClass = getattr(medmnist, INFO[flag]["python_class"])
        
        train_ds = DataClass(split="train", download=True)
        val_ds = DataClass(split="val", download=True)
        test_ds = DataClass(split="test", download=True)
        
        X = np.concatenate([train_ds.imgs, val_ds.imgs, test_ds.imgs], axis=0)
        y = np.concatenate([train_ds.labels, val_ds.labels, test_ds.labels], axis=0)
        
        X = tc.tensor(X, dtype=tc.float32)
        y = tc.tensor(y).long().squeeze()
        return X.permute(0, 3, 1, 2), y


    