
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
    figsize : Tuple, optional (default=[10, 8])
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
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[..., 0], X_pca[..., 1], X_pca[..., 2], c=y_batch, cmap="coolwarm", alpha=0.5)
        ax.set_zlabel("PC 3", fontsize=16)
    else:
        ax = fig.subplots()
        ax.scatter(X_pca[..., 0], X_pca[..., 1], c=y_batch, cmap="coolwarm", alpha=0.5)
        
    ax.set_title(f"PCA projection ({iter_size} batches)", fontsize=18)
    ax.set_xlabel("PC 1", fontsize=16)
    ax.set_ylabel("PC 2", fontsize=16)
    plt.tight_layout()
    plt.show();
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f'pca_{data}_{iter_size}_batches.png', bbox_inches='tight', dpi=dpi)

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
    figsize : Tuple, optional (default=[10, 8])
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
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_tsne[..., 0], X_tsne[..., 1], X_tsne[..., 2], c=y_batch, cmap='coolwarm', alpha=0.5)
        ax.set_zlabel("t-SNE 3", fontsize=16)
    else:
        ax = fig.subplots()
        ax.scatter(X_tsne[..., 0], X_tsne[..., 1], c=y_batch, cmap='coolwarm', alpha=0.5)

    ax.set_title(f"t-SNE projection ({iter_size} batches)", fontsize=18)
    ax.set_xlabel("t-SNE 1", fontsize=16)
    ax.set_ylabel("t-SNE 2", fontsize=16)
    plt.tight_layout()
    plt.show();
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f't_sne_{data}_{iter_size}_batches.png', bbox_inches='tight', dpi=dpi)




    


