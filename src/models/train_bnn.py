import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.BnnModel import BayesianModel
from src.data.data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)


def train_model(X_train, y_train, model, optimizer, loss_function, num_epochs):
    train_losses = []
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_function(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses


def train_model_variational(X_train, y_train, model, optimizer, num_epochs):
    train_losses = []
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        model.train()
        optimizer.zero_grad()
        loss = model.elbo_loss(X_train, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return model, train_losses


def plot_training_loss(train_losses, title="Training Loss Over Epochs"):
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_predictions(X_test, y_test, predictions, title="True Values vs Predictions"):
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_test, "b.", markersize=10, label="Ground Truth")
    plt.plot(X_test, predictions, "r.", markersize=10, label="Predictions")
    plt.xlabel("Input Features (X_test)")
    plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
    plt.title(title)
    plt.legend()
    plt.show()


def run_training(
    X_train, y_train, X_test, y_test, model, optimizer, loss_function, num_epochs
):
    train_losses = train_model(
        X_train, y_train, model, optimizer, loss_function, num_epochs
    )
    plot_training_loss(train_losses)

    with torch.no_grad():
        model.eval()
        predictions = model(X_test)
        predictions_np = predictions.numpy()

    plot_predictions(X_test, y_test, predictions_np)
    return model


MUANA_DATA_PATH = "data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "data/international-airline-passengers.csv"

# mauna_loa_atmospheric_co2 Dataset
X1, y1, X1_normalized = load_mauna_loa_atmospheric_co2(MUANA_DATA_PATH)
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1_normalized, y1, test_size=0.2, random_state=42
)
X1_train_tensor = torch.from_numpy(X1_train).float()
y1_train_tensor = torch.from_numpy(y1_train).float()
X1_test_tensor = torch.from_numpy(X1_test).float()

# international-airline-passengers Dataset
X2, y2, X2_normalized = load_international_airline_passengers(AIRLINE_DATA_PATH)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_normalized, y2, test_size=0.2, random_state=42
)
X2_train_tensor = torch.from_numpy(X2_train).float()
y2_train_tensor = torch.from_numpy(y2_train).float()
X2_test_tensor = torch.from_numpy(X2_test).float()

# Common model parameters
input_size = X1_train.shape[1]
hidden_size = 20
output_size = 1
num_epochs = 1000

# Training and evaluating models for mauna_loa_atmospheric_co2 Dataset
model_1 = BayesianModel(input_size, hidden_size, output_size, 5)
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.01)
loss_function = nn.MSELoss()

model_1 = run_training(
    X1_train_tensor,
    y1_train_tensor,
    X1_test_tensor,
    y1_test,
    model_1,
    optimizer_1,
    loss_function,
    num_epochs,
)
torch.save(model_1.state_dict(), "./models/mauna_loa_model.pth")

# Training and evaluating models for international-airline-passengers Dataset
model_2 = BayesianModel(input_size, hidden_size, output_size, 5)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.01)

model_2 = run_training(
    X2_train_tensor,
    y2_train_tensor,
    X2_test_tensor,
    y2_test,
    model_2,
    optimizer_2,
    loss_function,
    num_epochs,
)
torch.save(model_2.state_dict(), "./models/international_airline_passengers_model.pth")

# Training using variational inference for both datasets
model_1_var = BayesianModel(input_size, hidden_size, output_size, 5)
optimizer_1_var = torch.optim.Adam(model_1_var.parameters(), lr=0.01)
model_1_var, losses_1_var = train_model_variational(
    X1_train_tensor, y1_train_tensor, model_1_var, optimizer_1_var, num_epochs
)
plot_training_loss(losses_1_var)
torch.save(model_1_var.state_dict(), "./models/mauna_loa_model_var.pth")

model_2_var = BayesianModel(input_size, hidden_size, output_size, 5)
optimizer_2_var = torch.optim.Adam(model_2_var.parameters(), lr=0.01)
model_2_var, losses_2_var = train_model_variational(
    X2_train_tensor, y2_train_tensor, model_2_var, optimizer_2_var, num_epochs
)
plot_training_loss(losses_2_var)
torch.save(
    model_2_var.state_dict(), "./models/international_airline_passengers_model_var.pth"
)
