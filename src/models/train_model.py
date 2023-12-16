import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


from src.models.BnnModel import BayesianModel, DropoutBayesianModel, MLPBayesianModel
from src.data.data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)
from src.models.SimpleModel import Mlp

MUANA_DATA_PATH = "data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "data/international-airline-passengers.csv"


def train_test_model(model_class, dataset, m, layer_dim: list):

    # ------------------------------------------
    #  Dataset
    # ------------------------------------------

    # Prepare data
    if dataset == "muana":
        _, y1, X1_normalized = load_mauna_loa_atmospheric_co2(MUANA_DATA_PATH)
    elif dataset == "aireline":
        _, y1, X1_normalized = load_international_airline_passengers(AIRLINE_DATA_PATH)
    else:
        raise NameError("dataset not supported")
    # Split the data into training and test sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1_normalized, y1, test_size=0.2, random_state=42
    )

    # Convert NumPy arrays to PyTorch tensors
    X1_train_tensor = torch.from_numpy(X1_train).float()
    y1_train_tensor = torch.from_numpy(y1_train).float()
    X1_test_tensor = torch.from_numpy(X1_test).float()
    y1_test_tensor = torch.from_numpy(y1_test).float()

    # Define the Bayesian neural network model
    input_size = X1_train.shape[1]
    model = model_class(input_size, *layer_dim)

    # Define loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 1000
    train_losses = []
    argmin = num_epochs + 1
    t = 0
    test_losses = []
    argmin_test = num_epochs + 1
    t_test = 0
    with tqdm(range(num_epochs), unit="epochs") as pbar:
        for epoch in pbar:
            pbar.set_description(f"training {epoch+1}")
            # Forward pass
            outputs = model(X1_train_tensor)
            loss = loss_function(outputs, y1_train_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=train_losses[-1], min_loss=min(train_losses))
            with torch.no_grad():
                model.eval()
                predictions_1 = model(X1_test_tensor)
                loss = loss_function(predictions_1, y1_test_tensor)
                test_losses.append(loss.item())

            # early stopping
            t += 1
            if argmin != np.argmin(train_losses):
                argmin = np.argmin(train_losses)
                t = 0
            if t > 20:
                break

            t_test += 1
            if argmin_test != np.argmin(test_losses):
                argmin_test = np.argmin(test_losses)
                t_test = 0
            if t_test > 15:
                break

    # ---------  Plot training losses  ----------
    plt.plot(range(1, epoch + 2), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

    # ---------  Plot Ground Truth vs Predictions  ----------
    if m is not None:
        results = []
        for j in range(m):
            with torch.no_grad():
                predictions_1 = model(X1_test_tensor)
                results.append(predictions_1.squeeze().numpy())
        results = np.stack(results, dtype=np.float32)
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0)
    # Evaluate the model on the test set
    with torch.no_grad():
        model.eval()
        predictions_1 = model(X1_test_tensor)

    # Convert predictions to NumPy array for plotting
    predictions_np_1 = predictions_1.numpy()
    print(f"test score eval:{np.square(np.subtract(predictions_np_1,y1_test)).mean()}")
    print(f"test score mean:{np.square(np.subtract(mean,y1_test)).mean()}")
    # export model
    torch.save(model, "./models/" + dataset + "_model.pth")
    # print(X1_test.shape)
    # print(X1_test)
    plt.figure(figsize=(10, 6))
    plt.plot(X1_test, y1_test, "b.", markersize=10, label="Ground Truth")
    plt.plot(X1_test, mean, "r.", markersize=10, label="Predictions")
    plt.errorbar(X1_test, mean, fmt="og", yerr=std)
    plt.xlabel("Input Features (X_test)")
    plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
    plt.title("True Values vs Predictions")
    plt.legend()
    plt.show()


train_test_model(BayesianModel, "aireline", 100, [20, 1])
train_test_model(MLPBayesianModel, "aireline", 100, [100, 100, 1])
train_test_model(DropoutBayesianModel, "aireline", 100, [100, 100, 1])
train_test_model(Mlp, "aireline", 100, [100, 50, 1])
