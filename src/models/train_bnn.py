import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.distributions import Normal
import tqdm
from sklearn.model_selection import train_test_split
from .BnnModel import BayesianModel
from data.data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)

MUANA_DATA_PATH = "data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "data/international-airline-passengers.csv"

hidden_size = [20, 15]
output_size = 1
noise = 1.0
batch_size = 63
learning_rate = 0.08
num_epochs = 1500

# Dummy Test data
X_test = torch.tensor(np.linspace(-2, 2, 1000).reshape(-1, 1)).float()

# Train Loop function
def train_loop(dataloader, model, loss_fn, optimizer):
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

# Evaluation loop function
def evaluation_loop(model,X_test,num_epochs):
    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for i in tqdm.tqdm(range(num_epochs)):
            y_pred = model(X_test)
            y_pred_list.append(y_pred.detach().numpy())
        y_preds = np.concatenate(y_pred_list, axis=1)
        y_mean = np.mean(y_preds, axis=1)
        y_sigma = np.std(y_preds, axis=1)
    return y_mean,y_sigma


# Gaussian negative log likelihood function
def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    dist = Normal(loc=y_pred, scale=sigma)
    return torch.sum(-dist.log_prob(y_obs))


# Creating a Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, _features, target_feature):
        self._features = _features
        self.target_feature = target_feature

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):

        return self._features[idx], self.target_feature[idx]

# Plotting functions
# Plot training losses function
def plot_training_losses(train_losses):
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

#  Plot Ground Truth vs Predictions & Plot epistemic uncertainty
def plot_test(dataloader,data_test,y_mean,y_sigma):
    x_test = dataloader.dataset._features
    y_test = dataloader.dataset.target_feature
    plt.figure(figsize=(10, 6))
    plt.plot(data_test, y_mean, 'r-', label='Predictive mean')
    plt.scatter(x_test, y_test, marker='+', label='Test data')
    plt.fill_between(data_test.ravel(), 
                     y_mean + 2 * y_sigma, 
                     y_mean - 2 * y_sigma, 
                     alpha=0.5, label='Epistemic uncertainty')
    plt.xlabel("Input Features (X_test)")
    plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
    plt.title("True Values vs Predictions")
    plt.legend()
    plt.show()

# Runing function
def run(model,loss_function,train_dataloader,test_dataloader,X_test,num_epochs,learning_rate):
    # Define loss function and optimizer
    #loss_function = neg_log_likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    train_losses = []
    for t in tqdm.tqdm(range(num_epochs)):
        train_losses += train_loop(train_dataloader, model, loss_function, optimizer)
    print("Done!")
    # ---------  Plot training losses  ----------
    plot_training_losses(train_losses)
    # Evaluate the model on the test set
    y_mean,y_sigma = evaluation_loop(model,X_test,num_epochs)
    # ---------  Plot Ground Truth vs Predictions & Epistemic Uncertainty  ----------
    plot_test(test_dataloader,X_test,y_mean,y_sigma)


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
train_size,input_size = X1_train.shape
num_batches = train_size / batch_size
kl_weight = 1 / num_batches

# Define DataLoader Api for our dataset
train_dataloader_1 = DataLoader(
    CustomDataset(X1_train, y1_train), batch_size=batch_size, shuffle=True
)
test_dataloader_1 = DataLoader(
    CustomDataset(X1_test, y1_test), batch_size=batch_size, shuffle=True
)

model1 = BayesianModel(input_size, hidden_size, output_size, kl_weight, torch.relu)
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
# Hyperparameters
train_size,input_size = X2_train.shape
num_batches = train_size / batch_size
kl_weight = 1 / num_batches


# Define DataLoader Api for our dataset
train_dataloader_2 = DataLoader(
    CustomDataset(X2_train, y2_train), batch_size=batch_size, shuffle=True
)
test_dataloader_2 = DataLoader(
    CustomDataset(X2_test, y2_test), batch_size=batch_size, shuffle=True
)

model2 = BayesianModel(input_size, hidden_size, output_size, kl_weight, torch.relu)

