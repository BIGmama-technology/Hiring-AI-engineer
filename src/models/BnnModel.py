import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional as F

class BcnnLayer1D(nn.Module):
    """
    Bayesian Convolutional Neural Network Layer.

    Args:
        in_channels (int) : Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int) : Kernel size for the convolution.
        stride (int)      : Stride for the convolution.
        padding (int)     : Padding for the convolution.
        dilation (int)    : Dilation for the convolution.
        groups (int)      : Number of groups for the convolution.
        bias (bool)       : Whether to use bias in the convolution.

    Attributes:
        in_channels (int) : Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int) : Kernel size for the convolution.
        stride (int)      : Stride for the convolution.
        padding (int)     : Padding for the convolution.
        dilation (int)    : Dilation for the convolution.
        groups (int)      : Number of groups for the convolution.
        bias (bool)       : Whether to use bias in the convolution.
        weight_mu (nn.Parameter)  : Mean parameters for weight distribution.
        weight_std (nn.Parameter) : Standard deviation parameters for weight distribution.
        bias_mu (nn.Parameter)    : Mean parameters for bias distribution.
        bias_std (nn.Parameter)   : Standard deviation parameters for bias distribution.

    Methods:
        reset_parameters(): Initialize parameters with specific initialization schemes.
        forward(x): Forward pass through the layer.
        kl_layer

    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(BcnnLayer1D, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.groups       = groups
        self.bias         = bias

        # Parameters for the weight distribution
        self.weight_mu  = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.weight_std = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))

        # Parameters for the bias distribution
        if bias:
            self.bias_mu  = nn.Parameter(torch.Tensor(out_channels))
            self.bias_std = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias_mu  = None
            self.bias_std = None

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

        if self.bias_mu is not None:
            nn.init.zeros_(self.bias_mu)
            nn.init.constant_(self.bias_std, -5.0)

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the convolution.

        """
        # Sample weights and biases from normal distributions
        weight = Normal(self.weight_mu, torch.exp(self.weight_std)).rsample()
        if self.bias_mu is not None:
            bias = Normal(self.bias_mu, torch.exp(self.bias_std)).rsample()
        else:
            bias = None

        # Apply 1D convolution to input using sampled weights and biases
        return F.conv1d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        
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

    def __init__(self, input_size, hidden_size, output_size, kernel = 3):
        super(BayesianModel, self).__init__()
        self.kernel = kernel
        self.pads   = self.kernel-1
        if hidden_size%2 == 1 : 
            self.vars   = int((hidden_size-1)/2)
            self.layer1 = BnnLayer(input_size, self.vars)
            self.skip1d = BcnnLayer1D(input_size, self.vars+1, kernel_size = self.kernel, stride=1,padding=self.pads, dilation=2)
        else : 
            self.vars   = int(hidden_size/2)
            self.layer1 = BnnLayer(input_size, self.vars)
            self.skip1d = BcnnLayer1D(input_size, self.vars  , kernel_size = self.kernel, stride=1,padding=self.pads, dilation=2)
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
    
    def kl_net(self):
        '''
        a function to compute the KL Divergence between Preior and Posterior of all the layers of the layers
        return : 
            the kl divergence of the network layers
        '''
        kl = 0
        for m in [self.layer1, self.layer2, self.skip1d] :
            if isinstance(m, (BnnLayer, BcnnLayer1D)):
                b_prior = Normal(torch.zeros_like(m.bias_mu), torch.exp(m.bias_std))
                w_prior = Normal(torch.zeros_like(m.weight_mu), torch.exp(m.weight_std))
                
                w_posterior = Normal(m.weight_mu, torch.exp(m.weight_std))
                b_posterior = Normal(m.bias_mu, torch.exp(m.bias_std))
                
                kl_w = torch.distributions.kl.kl_divergence(w_posterior, w_prior).sum()
                kl_b = torch.distributions.kl.kl_divergence(b_posterior, b_prior).sum()
                
                kl += kl_w + kl_b
                
        return kl.detach()       