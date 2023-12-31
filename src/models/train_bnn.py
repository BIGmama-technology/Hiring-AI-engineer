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

MUANA_DATA_PATH = "data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "data/international-airline-passengers.csv"


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


# ------------------------------------------
# mauna_loa_atmospheric_co2 Dataset
# ------------------------------------------

X1, y1, X1_normalized = load_mauna_loa_atmospheric_co2(MUANA_DATA_PATH)
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1_normalized, y1, test_size=0.2, random_state=42
)

X1_train_tensor = torch.from_numpy(X1_train).float()
y1_train_tensor = torch.from_numpy(y1_train).float()
X1_test_tensor = torch.from_numpy(X1_test).float()

input_size = X1_train.shape[1]
hidden_size = 20
output_size = 1
model = BayesianModel(input_size, hidden_size, output_size, 5)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000

train_losses = train_model(
    X1_train_tensor, y1_train_tensor, model, optimizer, loss_function, num_epochs
)
plot_training_loss(train_losses)

with torch.no_grad():
    model.eval()
    predictions_1 = model(X1_test_tensor)
    predictions_np_1 = predictions_1.numpy()

plot_predictions(X1_test, y1_test, predictions_np_1)

torch.save(model, "./models/mauna_loa_model.pth")

# ------------------------------------------
# international-airline-passengers Dataset
# ------------------------------------------

X2, y2, X2_normalized = load_international_airline_passengers(AIRLINE_DATA_PATH)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_normalized, y2, test_size=0.2, random_state=42
)

X2_train_tensor = torch.from_numpy(X2_train).float()
y2_train_tensor = torch.from_numpy(y2_train).float()
X2_test_tensor = torch.from_numpy(X2_test).float()

input_size = X2_train.shape[1]
model = BayesianModel(input_size, hidden_size, output_size, num_layers=5)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses = train_model(
    X2_train_tensor, y2_train_tensor, model, optimizer, loss_function, num_epochs
)
plot_training_loss(train_losses)

with torch.no_grad():
    model.eval()
    predictions_2 = model(X2_test_tensor)
    predictions_np_2 = predictions_2.numpy()

plot_predictions(X2_test, y2_test, predictions_np_2)

torch.save(model, "./models/international_airline_passengers_model.pth")
