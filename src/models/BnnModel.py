import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional as F


class BnnLayer(nn.Module):
    """
    Bayesian Neural Network Layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        kl_weight (float): 1 / Nbr of batches (mini-batch gradient descent)
        prior_sigma_1 (float): variances of the mixturecomponents
        prior_sigma_2 (float): variances of the mixturecomponents
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

    def __init__(
        self,
        in_features,
        out_features,
        kl_weight,
        prior_sigma_1=1.5,
        prior_sigma_2=0.1,
        prior_pi=0.5,
        **kwargs
    ):

        super(BnnLayer, self).__init__()
        self.kl_weight = kl_weight
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.in_features = in_features
        self.out_features = out_features
        self.loss = None
        self.init_sigma = np.sqrt(
            self.prior_pi_1 * self.prior_sigma_1**2
            + (1 - self.prior_pi_1) * self.prior_sigma_2**2
        )
        # Softplus function
        self.softplus = nn.Softplus()

        # Parameters for the weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_std = nn.Parameter(torch.Tensor(out_features, in_features))

        # Parameters for the bias distribution
        self.bias_mu = nn.Parameter(
            torch.Tensor(
                out_features,
            )
        )
        self.bias_std = nn.Parameter(
            torch.Tensor(
                out_features,
            )
        )

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.normal_(self.weight_mu, std=self.init_sigma)
        nn.init.normal_(self.weight_std, std=self.init_sigma)

        nn.init.zeros_(self.bias_mu)
        nn.init.zeros_(self.bias_std)

    def forward(self, x):
        kernel_sigma = self.softplus(self.weight_std)
        kernel = self.weight_mu + kernel_sigma * torch.randn(self.weight_mu.shape)

        bias_sigma = self.softplus(self.bias_std)
        bias = self.bias_mu + bias_sigma * torch.randn(self.bias_mu.shape)
        # Add layer-wise complexity cost to the total loss
        self.loss = self.kl_loss(kernel, self.weight_mu, kernel_sigma) + self.kl_loss(
            bias, self.bias_mu, bias_sigma
        )

        return F.linear(x, kernel, bias)

    def log_prior_prob(self, w):
        comp_1_dist = Normal(0.0, self.prior_sigma_1)
        comp_2_dist = Normal(0.0, self.prior_sigma_2)
        return torch.log(
            self.prior_pi_1 * comp_1_dist.log_prob(w).exp()
            + (1 - self.prior_pi_1) * comp_2_dist.log_prob(w).exp()
        )

    # Compute layer-wise complexity cost
    def kl_loss(self, w, mu, sigma):
        variational_dist = Normal(mu, sigma)
        return self.kl_weight * torch.sum(
            variational_dist.log_prob(w) - self.log_prior_prob(w)
        )


class BayesianModel(nn.Module):
    """
    Bayesian Neural Network Model.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Vector of hidden units in the middle layers.
        output_size (int): Number of output features.
        kl_weight (float) : 1 / Nbr of batches (mini-batch gradient descent)
        activation : Activation function.

    Attributes:
        layer1 (BnnLayer): First Bayesian Neural Network layer.
        layer2 (BnnLayer): Second Bayesian Neural Network layer.
        layer3 (BnnLayer): Third Bayesian Neural Network layer.

    Methods:
        forward(x): Forward pass through the model.

    """

    def __init__(self, input_size, hidden_size, output_size, kl_weight, activation):
        super(BayesianModel, self).__init__()
        self.activation = activation
        self.loss = 0.0
        # Define the architecture of the Bayesian neural network
        self.layer1 = BnnLayer(input_size, hidden_size[0], kl_weight)
        self.layer2 = BnnLayer(hidden_size[0], hidden_size[1], kl_weight)
        self.layer3 = BnnLayer(hidden_size[1], output_size, kl_weight)

    # Calculate the sum of layer-wise complexity costs of all layers
    def get_losses_layers(self):
        losses = []
        for layer in self.children():
            losses.append(layer.loss)
        self.loss = torch.stack(losses).sum()

    def forward(self, x):
        # Apply activation function to the output of the first layer
        x = self.activation(self.layer1(x))
        # Pass the result through the second layer
        x = self.activation(self.layer2(x))
        # Pass the result through the third layer
        x = self.layer3(x)
        # calculate sum of layers's loss
        self.get_losses_layers()
        return x