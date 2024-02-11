import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

    # Convert NumPy arrays to PyTorch tensors
    X1_train_tensor = torch.from_numpy(X1_train).float()
    y1_train_tensor = torch.from_numpy(y1_train).float()
    X1_test_tensor = torch.from_numpy(X1_test).float()

    # Define the Bayesian neural network model
    input_size = X1_train.shape[1]
    hidden_size = 20
    output_size = 1
    model = BayesianModel(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 1000
    train_losses = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X1_train_tensor)
        loss = loss_function(outputs, y1_train_tensor)

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

    # Evaluate the model on the test set
    with torch.no_grad():
        model.eval()
        predictions_1 = model(X1_test_tensor)

    # Convert predictions to NumPy array for plotting
    predictions_np_1 = predictions_1.numpy()

    # export model
    torch.save(model, "./models/mauna_loa_model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(X1_test, y1_test, "b.", markersize=10, label="Ground Truth")
    plt.plot(X1_test, predictions_1, "r.", markersize=10, label="Predictions")
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

    # Convert NumPy arrays to PyTorch tensors
    X2_train_tensor = torch.from_numpy(X2_train).float()
    y2_train_tensor = torch.from_numpy(y2_train).float()
    X2_test_tensor = torch.from_numpy(X2_test).float()

    # Define the Bayesian neural network model
    input_size = X2_train.shape[1]
    hidden_size = 20
    output_size = 1
    model = BayesianModel(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 1000
    train_losses = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X2_train_tensor)
        loss = loss_function(outputs, y2_train_tensor)

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

    # Evaluate the model on the test set
    with torch.no_grad():
        model.eval()
        predictions_2 = model(X2_test_tensor)

    # Convert predictions to NumPy array for plotting
    predictions_np_2 = predictions_2.numpy()

    # export model
    torch.save(model, "./models/international_airline_passengers_model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(X2_test, y2_test, "b.", markersize=10, label="Ground Truth")
    plt.plot(X2_test, predictions_2, "r.", markersize=10, label="Predictions")
    plt.xlabel("Input Features (X_test)")
    plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
    plt.title("True Values vs Predictions")
    plt.legend()
    plt.show()
