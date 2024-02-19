"""
This module contains the implementation of an improved Bayesian Neural Network (BNN) model
"""
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class BnnLayer(nn.Module):
    """
    Bayesian Neural Network Layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        batch_norm (bool): Flag to enable batch normalization.


    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        weight_mu (nn.Parameter): Mean parameters for weight distribution.
        weight_std (nn.Parameter): Standard deviation parameters for weight distribution.
        bias_mu (nn.Parameter): Mean parameters for bias distribution.
        bias_std (nn.Parameter): Standard deviation parameters for bias distribution.
        batch_norm (bool): Flag indicating whether batch normalization is used.

    Methods:
        reset_parameters(): Initialize parameters with specific initialization schemes.
        forward(x): Forward pass through the layer.

    """

    def __init__(self, in_features, out_features, batch_norm=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.batch_norm = batch_norm

        # Parameters for the weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_std = nn.Parameter(torch.Tensor(out_features, in_features))

        # Parameters for the bias distribution
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_std = nn.Parameter(torch.Tensor(out_features))

        if self.batch_norm:
            # Parameters for batch normalization
            self.bn = nn.BatchNorm1d(out_features)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weight and bias parameters with specific initialization schemes.
        Weight parameters are initialized using Kaiming normal initialization,
        and standard deviation parameters are initialized with a constant value.
        Bias parameters are initialized with zeros, and bias standard deviation
        parameters are initialized with a constant value.
        """
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.weight_std, -4.0)

        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_std, -4.0)

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the linear transformation.

        """
        # Sample weights and biases from normal distributions
        weight = Normal(self.weight_mu, torch.exp(self.weight_std)).rsample()
        bias = Normal(self.bias_mu, torch.exp(self.bias_std)).rsample()

        # Apply linear transformation to input using sampled weights and biases
        x = F.linear(x, weight, bias)

        if self.batch_norm:
            # Apply batch normalization
            x = self.bn(x)

        return x


class BayesianModel(nn.Module):
    """
    Bayesian Neural Network Model.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in the middle layer.
        output_size (int): Number of output features.

    Attributes:
        layer1 (BnnLayer): First Bayesian Neural Network layer.
        layer2 (BnnLayer): Second Bayesian Neural Network layer.

    Methods:
        forward(x): Forward pass through the model.

    """

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()

        # Define the architecture of the Bayesian neural network
        self.layer1 = BnnLayer(input_size, hidden_size1)
        self.layer2 = BnnLayer(hidden_size1, hidden_size2)
        self.layer3 = BnnLayer(hidden_size2, output_size)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the model.

        """
        # Apply ReLU activation to the output of the first layer
        x = F.leaky_relu(self.layer1(x))
        # Pass the result through the second layer
        x = F.leaky_relu(self.layer2(x))
        # Pass the result through the third layer
        x = self.layer3(x)
        return x

    def predict(self, x):
        """
        Desc:
            Sample predictions from the posterior distribution of the model.
        Args:
            x (array-like): Input features.
        Returns:
            tensor: Predicted values.
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions
