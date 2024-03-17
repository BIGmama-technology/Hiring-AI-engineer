import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.distributions import Normal
from sklearn.model_selection import train_test_split
from .BnnModel import BayesianModel
from data.data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)

MUANA_DATA_PATH = "data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "data/international-airline-passengers.csv"

# ------------------------------------------
# mauna_loa_atmospheric_co2 Dataset
# ------------------------------------------

# Prepare data
X1, y1, X1_normalized = load_mauna_loa_atmospheric_co2(MUANA_DATA_PATH)

# Split the data into training and test sets
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1_normalized, y1, test_size=0.2, random_state=42
)

# Convert NumPy arrays to PyTorch tensors
X1_train_tensor = torch.from_numpy(X1_train).float()
y1_train_tensor = torch.from_numpy(y1_train).float()
X1_test_tensor = torch.from_numpy(X1_test).float()

# Hyperparameters
input_size = X1_train.shape[1]
hidden_size = [20, 15]
output_size = 1
noise = 1.0
train_size = X1_train.shape[0]
batch_size = 63
num_batches = train_size / batch_size
kl_weight = 1 / num_batches
learning_rate = 0.08
num_epochs = 1500


# Creating a Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, _features, target_feature):
        self._features = _features
        self.target_feature = target_feature

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):

        return self._features[idx], self.target_feature[idx]


# Define DataLoader Api for our dataset
train_dataloader = DataLoader(
    CustomDataset(X1_train, y1_train), batch_size=batch_size, shuffle=True
)
test_dataloader = DataLoader(
    CustomDataset(X1_test, y1_test), batch_size=batch_size, shuffle=True
)

# Define train Loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_losses = []
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.float())
        loss = loss_fn(pred, y) + model.loss
        train_losses.append(loss.item())
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return train_losses


# Gaussian negative log likelihood function
def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    dist = Normal(loc=y_pred, scale=sigma)
    return torch.sum(-dist.log_prob(y_obs))