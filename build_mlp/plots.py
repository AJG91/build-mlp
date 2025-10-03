import numpy as np
import pandas as pd
import torch as tc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Tuple

def plot_pca(
    loader: tc.utils.data.DataLoader, 
    path: str, 
    dpi: int, 
    data: str,
    n_components: int, 
    iter_size: int = 100, 
    figsize: Tuple = (10, 8)
) -> None:
    """
    Plots the PCA of a subset of a given dataset.

    Parameters
    ----------
    loader : tc.utils.data.DataLoader
        DataLoader for the training set.
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    data : str
        The name of the dataset being plotted.
        This is used to label the plot when saving to directory.
    iter_size : int, optional (default=100)
        Specifies the number of batch iterations to plot.
    figsize : Tuple, optional (default=(10, 8))
        Specifies the figure size.
    """
    X_batch, y_batch = [], []
    for _ in range(iter_size):
        try:
            X_i, y_i = next(iter(loader))
            X_batch.append(X_i)
            y_batch.append(y_i)
        except StopIteration:
            break

    X_batch = tc.concat(X_batch, dim=0).numpy()
    y_batch = tc.concat(y_batch, dim=0).numpy().ravel()

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_batch)
    
    fig = plt.figure(figsize=figsize)
    
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_pca[..., 0], X_pca[..., 1], X_pca[..., 2], c=y_batch, cmap="coolwarm", alpha=0.5)
        ax.set_zlabel("PC 3", fontsize=16)
    else:
        ax = fig.subplots()
        ax.scatter(X_pca[..., 0], X_pca[..., 1], c=y_batch, cmap="coolwarm", alpha=0.5)
        
    ax.set_title(f"PCA projection ({iter_size} batches)", fontsize=18)
    ax.set_xlabel("PC 1", fontsize=16)
    ax.set_ylabel("PC 2", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"pca_{data}_{iter_size}_batches.png", bbox_inches="tight", dpi=dpi)

def plot_t_sne(
    loader: tc.utils.data.DataLoader, 
    path: str, 
    dpi: int, 
    data: str,
    n_components: int, 
    iter_size: int = 100, 
    figsize: Tuple = (10, 8)
) -> None:
    """
    Plots the t-SNE of a subset of a given dataset.

    Parameters
    ----------
    loader : tc.utils.data.DataLoader
        DataLoader for the training set.
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    data : str
        The name of the dataset being plotted.
        This is used to label the plot when saving to directory.
    iter_size : int, optional (default=100)
        Specifies the number of batch iterations to plot.
    figsize : Tuple, optional (default=(10, 8))
        Specifies the figure size.
    """    
    X_batch, y_batch = [], []
    for _ in range(iter_size):
        try:
            X_i, y_i = next(iter(loader))
            X_batch.append(X_i)
            y_batch.append(y_i)
        except StopIteration:
            break

    X_batch = tc.concat(X_batch, dim=0).numpy()
    y_batch = tc.concat(y_batch, dim=0).numpy().ravel()
    
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_batch)

    fig = plt.figure(figsize=figsize)
    
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_tsne[..., 0], X_tsne[..., 1], X_tsne[..., 2], c=y_batch, cmap="coolwarm", alpha=0.5)
        ax.set_zlabel("t-SNE 3", fontsize=16)
    else:
        ax = fig.subplots()
        ax.scatter(X_tsne[..., 0], X_tsne[..., 1], c=y_batch, cmap="coolwarm", alpha=0.5)

    ax.set_title(f"t-SNE projection ({iter_size} batches)", fontsize=18)
    ax.set_xlabel("t-SNE 1", fontsize=16)
    ax.set_ylabel("t-SNE 2", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"t_sne_{data}_{iter_size}_batches.png", bbox_inches="tight", dpi=dpi)


def plot_feature_distributions(
    df: pd.DataFrame, 
    path: str, 
    dpi: int, 
    data: str,
    bins: int = 30, 
    figsize: Tuple = (15, 10)
) -> None:
    """
    Plots histograms of the different features.
    Outputs a 3x3 plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    data : str
        The name of the dataset being plotted.
        This is used to label the plot when saving to directory.
    bins : int, optional (default=30)
        Specifies the number of bins in distribution.
    figsize : Tuple, optional (default=(15, 10))
        Specifies the figure size.
    """
    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=figsize, constrained_layout=True
    )
    df.hist(bins=bins, ax=axes, grid=False)
    for ax in axes.ravel():
        ax.set_title(ax.get_title(), fontsize=14)
        
    plt.show()
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"feature_distribution_{data}.png", bbox_inches="tight", dpi=dpi)

def plot_price_by_location(
    df: pd.DataFrame, 
    path: str, 
    dpi: int, 
    data: str,
    figsize: Tuple = (10, 8)
) -> None:
    """
    Plots latitude-longitude heat map of California based on housing prices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    data : str
        The name of the dataset being plotted.
        This is used to label the plot when saving to directory.
    figsize : Tuple, optional (default=(10, 8))
        Specifies the figure size.
    """
    fig = plt.figure(figsize=figsize)
    
    sc = plt.scatter(
        df["Longitude"], df["Latitude"], c=df["MedHouseVal"],
        cmap="viridis", s=20, alpha=0.5
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Median House Value", fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)
    plt.show()
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"price_by_location_{data}.png", bbox_inches="tight", dpi=dpi)

def plot_population_and_prices(
    df: pd.DataFrame, 
    path: str, 
    dpi: int, 
    data: str,
    figsize: Tuple = (10, 8)
) -> None:
    """
    Plots latitude-longitude heat map of California based on population and housing prices. 
    The size of the circle is proportional to the total population in the area.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    data : str
        The name of the dataset being plotted.
        This is used to label the plot when saving to directory.
    figsize : Tuple, optional (default=(10, 8))
        Specifies the figure size.
    """
    fig = plt.figure(figsize=figsize)
    
    sc = plt.scatter(
        df["Longitude"], df["Latitude"], 
        alpha=0.4, s=df["Population"]/100,
        c=df["MedHouseVal"], cmap="plasma"
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Median House Value", fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)
    plt.show()
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"price_and_population_{data}.png", bbox_inches="tight", dpi=dpi)

def plot_correlation_map(
    df: pd.DataFrame, 
    path: str, 
    dpi: int, 
    data: str,
    figsize: Tuple = (10, 8)
) -> None:
    """
    Plots the correlation between fetures.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    data : str
        The name of the dataset being plotted.
        This is used to label the plot when saving to directory.
    figsize : Tuple, optional (default=(10, 8))
        Specifies the figure size.
    """
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(corr, cmap="coolwarm")
    cbar = plt.colorbar(cax)
    cbar.set_label("Correlation", fontsize=14)
    
    for (i, j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
    
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    plt.show()
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"correlation_map_{data}.png", bbox_inches="tight", dpi=dpi)

def plot_metrics_vs_epochs(
    train_metric: np.ndarray, 
    val_metric: np.ndarray, 
    epochs: int, 
    path: str, 
    dpi: int, 
    metric: str, 
    figsize: Tuple = (10, 8)
) -> None:
    """
    Plots the training and validation metrics over total epochs.

    Parameters
    ----------
    train_metric : tc.Tensor
        Metrics generated during the training phase.
        Example: loss, MSE, R2-score
    val_metric : tc.Tensor
        Metrics generated during the validation phase.
        Example: loss, MSE, R2-score
    epochs : int
        Number of epochs used to train model.
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    figsize : Tuple, optional (default=(10, 8))
        Specifies the figure size.
    """
    epochs_mesh = tc.arange(0, epochs)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs_mesh, train_metric, label="training set")
    ax.plot(epochs_mesh, val_metric, label="validation set")

    ax.set_xlabel("epochs", fontsize=14)
    ax.set_ylabel(f"{metric}", fontsize=14)
    ax.legend(fontsize=14)
    plt.show()
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f"{metric}_vs_epochs.png", bbox_inches="tight", dpi=dpi)