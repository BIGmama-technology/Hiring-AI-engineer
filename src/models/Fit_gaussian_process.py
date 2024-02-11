import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# import data_loader

from src.models.GaussianProcess import *
from src.models.kernels import *
from src.models.GaussianProcess import *
from src.data import data_loader

# from src.data import dataloader
from src.data.data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)
import sys
import os

os.system("clear")
MUANA_DATA_PATH = "data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "data/international-airline-passengers.csv"

# ------------------------------------------
# mauna_loa_atmospheric_co2 Dataset
# ------------------------------------------
def fit_pgr_train_bnn_mauna_loa_atmospheric_co2():

    # Prepare data
    X1, y1, X1_normalized = load_mauna_loa_atmospheric_co2(MUANA_DATA_PATH)
    X1_train, y1_train = X1_normalized, y1

    # Creating test data
    x_test = np.linspace(X1_train.min(), X1_train.max(), 50).reshape(-1, 1)

    # genirating GaussianProcess
    GP = GaussianProcess(GaussianKernel())

    # putting data into model (training)
    GP.fit(X1_train, y1_train)

    # make prediction  on test data
    mu_post, sd_post = GP.predict(x_test)

    # plotting result
    plt.figure(figsize=(10, 6))
    plt.plot(X1_train, y1_train, "b.", markersize=10, label="Ground Truth")
    plt.plot(x_test, mu_post, "r.", markersize=10, label="Predictions")
    plt.xlabel("Input Features (X_test)")
    plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
    plt.title("True Values vs Predictions")
    plt.legend()
    plt.show()


# ------------------------------------------
# international-airline-passengers Dataset
# ------------------------------------------
def fit_pgr_train_bnn_international_airline_passengers():
    # Prepare data
    X2, y2, X2_normalized = load_international_airline_passengers(AIRLINE_DATA_PATH)
    X2_train, y2_train = X2_normalized, y2

    # Creating test data
    x_test = np.linspace(X2_train.min(), X2_train.max(), 50).reshape(-1, 1)

    # genirating GaussianProcess
    GP = GaussianProcess(GaussianKernel())

    # putting data into model (training)
    GP.fit(X2_train, y2_train)

    # make prediction  on test data
    mu_post2, sd_post2 = GP.predict(x_test)

    # plotting result
    plt.figure(figsize=(10, 6))
    plt.plot(X2_train, y2_train, "b.", markersize=10, label="Ground Truth")
    plt.plot(x_test, mu_post2, "r.", markersize=10, label="Predictions")
    plt.xlabel("Input Features (X_test)")
    plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
    plt.title("True Values vs Predictions")
    plt.legend()
    plt.show()
