import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.models.BnnModel import BayesianModel, BcnnLayer1D, BnnLayer
from src.models.refactors import Refactor, Refactor_var

from src.data.data_loader import (
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
input_size = X1_train.shape[1]
hidden_size = 20
output_size = 1
model = BayesianModel(input_size, hidden_size, output_size, kernel=3)
# ------------------------------------------
# Train the model with Loss Function MAE
# ------------------------------------------
# Define loss function and optimizer
loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

fact1 = Refactor( [X1_train_tensor, y1_train_tensor], 
                    [X1_test_tensor, y1_test],
                    model = model, criterion = loss_function, 
                    optimizer = optimizer,
                    epochs = 1000)

fact1.fit()
fact1.plot_loss()
fact1.eval()
fact1.save_model(path = "./models/", name = "international_airline_passengers_model")

# --------------------------------------------
# Train the model with ELBO Loss using Var Inf.
# ---------------------------------------------

var_fact1 = Refactor_var( [X1_train_tensor, y1_train_tensor], 
                      [X1_test_tensor , y1_test],
                       model = model,
                       optimizer = optimizer,
                       epochs = 1000)

var_fact1.fit()
var_fact1.plot_loss()
var_fact1.eval(plot=True)
var_fact1.save_model(path = "./models/", name = "VAR_mauna_loa_model")

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
hidden_size = 20
output_size = 1
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ------------------------------------------
# Train the model with Loss Function MAE
# ------------------------------------------
loss_function = nn.L1Loss()
fact2 =  Refactor([X2_train_tensor, y2_train_tensor], 
                    [X2_test_tensor, y2_test],
                    model = model, criterion = loss_function, 
                    optimizer = optimizer,
                    epochs = 1000)
fact2.fit()
fact2.plot_loss()
fact2.eval()
fact2.save_model(path = "./models/", name = "international_airline_passengers_model")


# --------------------------------------------
# Train the model with ELBO Loss using Var Inf.
# ---------------------------------------------

var_fact2 = Refactor_var( [X2_train_tensor, y2_train_tensor], 
                          [X2_test_tensor , y2_test],
                           model = model,
                           optimizer = optimizer,
                           epochs = 1000)

var_fact2.fit()
var_fact2.plot_loss()
var_fact2.eval(plot=True)
var_fact2.save_model(path = "./models/", name = "VAR_international_airline_passengers_model")
