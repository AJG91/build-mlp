import torch as tc
import torch.nn as nn

class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) for regression.
    Inherits functionality from nn.Module.

    Architecture:
        - Linear -> ReLU -> Dropout
        - Linear -> ReLU
        - Linear (output)
        
    Attributes
    ----------
    net : nn.Sequential
        Multi-layer perceptron.

    Parameters
    ----------
    input_dim : int
        Dimension of input data.
    hidden_dim : int
        Number of neurons in the hidden fully connected layer.
    dropout : float
        Indicates the fraction of neurons deactivated during training.
        Decimal between 0 and 1.
    """
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Parameters
        ----------
        x : tc.Tensor
            Input tensor of shape (N, *) where N is the batch size.

        Returns
        -------
        tc.Tensor
            Output tensor produced by the network after the forward pass.
        """
        return self.net(x)
