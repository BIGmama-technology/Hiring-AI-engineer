import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional as F


class BnnLayer(nn.Module):
    """
    Bayesian Neural Network Layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        weight_mu (nn.Parameter): Mean parameters for weight distribution.
        weight_std (nn.Parameter): Standard deviation parameters for weight distribution.
        bias_mu (nn.Parameter): Mean parameters for bias distribution.
        bias_std (nn.Parameter): Standard deviation parameters for bias distribution.

    Methods:
        reset_parameters(): Initialize parameters with specific initialization schemes.
        forward(x): Forward pass through the layer.

    """

    def __init__(self, in_features, out_features):
        super(BnnLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Parameters for the weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_std = nn.Parameter(torch.Tensor(out_features, in_features))

        # Parameters for the bias distribution
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_std = nn.Parameter(torch.Tensor(out_features))

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
        nn.init.constant_(self.weight_std, -5.0)

        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_std, -5.0)

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
        return F.linear(x, weight, bias)



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

    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianModel, self).__init__()
        if hidden_size%2 == 1 : 
            self.vars   = int((hidden_size-1)/2)
            self.layer1 = BnnLayer(input_size, self.vars)
            self.skip1d = torch.nn.Conv1d(1, self.vars+1, 1, stride=1,dilation=2)
        else : 
            self.vars   = int(hidden_size/2)
            self.layer1 = BnnLayer(input_size, self.vars)
            self.skip1d = torch.nn.Conv1d(1, self.vars, 1, stride=1,dilation=2)
        self.layer2 = BnnLayer(hidden_size, output_size)
        

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the model.

        """
        # Apply Conv1D for the input data (expand dims, Conv1D, squeeze) 
        inp = x[...,None]
        skiped = self.skip1d(inp).squeeze()
        
        # Apply ReLU activation to the output of the first layer
        x = F.relu(self.layer1(x))
        #concatenate the linear data with the convoluted input ( Concat is better then add) 
        x = torch.cat([x,skiped], -1)
        # Pass the result through the second layer
        x = self.layer2(x)
        return x
