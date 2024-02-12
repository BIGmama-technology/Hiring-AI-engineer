import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# import tqdm
from tqdm import tqdm
import numpy as np
from src.models.BnnModel import BayesianModel
from src.data.data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)

MUANA_DATA_PATH = "data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "data/international-airline-passengers.csv"

# ------------------------------------------
# mauna_loa_atmospheric_co2 Dataset
# ------------------------------------------
def train_bnn_mauna_loa_atmospheric_co2():
    # Prepare data
    X1, y1, X1_normalized = load_mauna_loa_atmospheric_co2(MUANA_DATA_PATH)

    # Split the data into training and test sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1_normalized, y1, test_size=0.2, random_state=42
    )

    # test data sorting and flattening ( for Plotting  )

    y1_test = y1_test.ravel()
    indices = np.argsort(X1_test.ravel())
    X1_test = X1_test[indices]
    y1_test = y1_test[indices]

    # Convert NumPy arrays to PyTorch tensors
    X1_train_tensor = torch.from_numpy(X1_train).float()
    y1_train_tensor = torch.from_numpy(y1_train).float()
    X1_test_tensor = torch.from_numpy(X1_test).float()

    # Define the Bayesian neural network model
    input_size = X1_train.shape[1]
    hidden_size1 = 12
    hidden_size2 = 8
    output_size = 1
    model = BayesianModel(input_size, hidden_size1, hidden_size2, output_size)

    # Define loss function and optimizer
    def loss_function(outputs, target, kl_divergence):
        mse = nn.MSELoss()
        return mse(outputs, target) + 0.5 * kl_divergence

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 1000
    train_losses = []

    for epoch in tqdm(range(num_epochs)):
        # Forward pass
        outputs, kl_divergence = model(X1_train_tensor)
        loss = loss_function(outputs, y1_train_tensor, kl_divergence)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # ---------  Plot training losses  ----------

    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

    # ---------  Plot Ground Truth vs Predictions  ----------

    # Evaluate the models on the test set
    with torch.no_grad():
        model.eval()
        predictions_1 = torch.cat(
            [model(X1_test_tensor)[0] for _ in range(1000)], dim=1
        )

    # extract the mean and std of my models prediction , and convert them  to numpy
    mu = predictions_1.numpy().mean(axis=1).ravel()
    y_std = predictions_1.numpy().std(axis=1).ravel()

    # export model
    torch.save(model, "./models/mauna_loa_model.pth")

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(X1_test, y1_test, "b.", markersize=10, label="Ground Truth")
    plt.plot(X1_test, mu, "r.", markersize=10, label="Predictions")
    plt.fill_between(X1_test.ravel(), mu - 10 * y_std, mu + 10 * y_std)
    plt.xlabel("Input Features (X_test)")
    plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
    plt.title("True Values vs Predictions")
    plt.legend()
    plt.show()


# ------------------------------------------
# international-airline-passengers Dataset
# ------------------------------------------
def train_bnn_international_airline_passengers():
    # Prepare data
    X2, y2, X2_normalized = load_international_airline_passengers(AIRLINE_DATA_PATH)

    # Split the data into training and test sets
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2_normalized, y2, test_size=0.2, random_state=42
    )

    # test data sorting and flattening  ( for Plotting  )
    y2_test = y2_test.ravel()
    indices = np.argsort(X2_test.ravel())
    X2_test = X2_test[indices]
    y2_test = y2_test[indices]

    # Convert NumPy arrays to PyTorch tensors
    X2_train_tensor = torch.from_numpy(X2_train).float()
    y2_train_tensor = torch.from_numpy(y2_train).float()
    X2_test_tensor = torch.from_numpy(X2_test).float()

    # Define the Bayesian neural network model
    input_size = X2_train.shape[1]
    hidden_size1 = 10
    hidden_size2 = 5
    output_size = 1
    model = BayesianModel(input_size, hidden_size1, hidden_size2, output_size)

    # Define loss function and optimizer
    def loss_function(outputs, target, kl_divergence):
        mse = nn.MSELoss()
        return mse(outputs, target) + 0.5 * kl_divergence

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 1000
    train_losses = []

    for epoch in tqdm(range(num_epochs)):
        # Forward pass
        outputs, kl_divergence = model(X2_train_tensor)
        loss = loss_function(outputs, y2_train_tensor, kl_divergence)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # ---------  Plot training losses  ----------

    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

    # ---------  Plot Ground Truth vs Predictions  ----------

    # Evaluate the models on the test set
    with torch.no_grad():
        model.eval()
        predictions_2 = torch.cat(
            [model(X2_test_tensor)[0] for _ in range(1000)], dim=1
        )

    # extract the mean and std of my models prediction , and convert them  to numpy
    mu = predictions_2.numpy().mean(axis=1).ravel()
    y_std = predictions_2.numpy().std(axis=1).ravel()

    # export model
    torch.save(model, "./models/international_airline_passengers_model.pth")

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(X2_test, y2_test, "b.", markersize=10, label="Ground Truth")
    plt.plot(X2_test, mu, "r.", markersize=10, label="Predictions")
    plt.fill_between(X2_test.ravel(), mu - 10 * y_std, mu + 10 * y_std)
    plt.xlabel("Input Features (X_test)")
    plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
    plt.title("True Values vs Predictions")
    plt.legend()
    plt.show()
