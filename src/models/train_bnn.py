"""
    This script trains a Bayesian neural network (BNN) model
    on the international-airline-passengers dataset and the
    mauna_loa_atmospheric_co2 dataset.
"""
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_mauna_loa_atmospheric_co2
from src.data.data_loader import load_international_airline_passengers


from src.data.data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)
from sklearn.model_selection import train_test_split

from src.models.BnnModel import BayesianModel


MUANA_DATA_PATH = "./data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "./data/international-airline-passengers.csv"


def train_bnn_model(
    model, x_train_tensor, y_train_tensor, num_epochs=1000, lr=0.01, batch_size=32
):
    """
    Train a Bayesian Neural Network model.

    Parameters:
    - model (BayesianModel): The Bayesian Neural Network model to be trained.
    - x_train_tensor (torch.Tensor): The input training data tensor.
    - y_train_tensor (torch.Tensor): The target training data tensor.
    - num_epochs (int): The number of training epochs (default is 1000).
    - lr (float): The learning rate for the optimizer (default is 0.01).
    - batch_size (int): Batch size for training (default is 32).

    Returns:
    List[float]: A list of training losses for each epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_losses = []

    with tqdm(total=num_epochs, desc="Training", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                # Forward pass
                outputs = model(inputs)
                loss = loss_function(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            average_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(average_epoch_loss)
            pbar.set_postfix({"Loss": average_epoch_loss})
            pbar.update()

    return train_losses


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

# Define the Bayesian neural network model
input_size = X1_train.shape[1]
HIDDEN_SIZE1 = 20
HIDDEN_SIZE2 = 40
OUTPUT_SIZE = 1
model_mauna = BayesianModel(input_size, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)


# Training loop
epochs = 1000
train_losses_mauna = train_bnn_model(model_mauna, X1_train_tensor, y1_train_tensor)

# export the model
torch.save(model_mauna, "./models/bnn_mauna.pth")

# ------------------------------------------
# international-airline-passengers Dataset
# ------------------------------------------
# Prepare data
X2, y2, X2_normalized = load_international_airline_passengers(AIRLINE_DATA_PATH)

# Split the data into training and test sets
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_normalized, y2, test_size=0.2, random_state=42
)

# Convert NumPy arrays to PyTorch tensors
X2_train_tensor = torch.from_numpy(X2_train).float()
y2_train_tensor = torch.from_numpy(y2_train).float()
X2_test_tensor = torch.from_numpy(X2_test).float()

# Define the Bayesian neural network model
input_size = X2_train.shape[1]
HIDDEN_SIZE1 = 10
HIDDEN_SIZE2 = 40
OUTPUT_SIZE = 1
model_airline = BayesianModel(input_size, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)


# Training loop
train_losses_airline = train_bnn_model(model_airline, X2_train_tensor, y2_train_tensor)

# export the model
torch.save(model_airline, "./models/bnn_airline.pth")
