# eliasboulham@gmail.com
from src.models.BnnModel import *
from src.models.train_bnn import *
from src.models.BnnModel import *
from src.models.Fit_gaussian_process import *

import torch
import numpy as np
from tabulate import tabulate
import torch.nn as nn
from src.data.data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)

os.system("clear")
print("from main")


def test():
    print("Choose a method :")
    print("1. BNN")
    print("2. GPR")
    choice = input("Enter the number of your choice: ")

    if choice == "1":
        print("Choose a dataset :")
        print("1. mauna_loa_atmospheric_co2")
        print("2. international_airline_passengers")
        choice = input("Enter the number of your choice: ")
        if choice == "1":
            train_bnn_mauna_loa_atmospheric_co2()
        elif choice == "2":
            train_bnn_international_airline_passengers()
        else:
            print("Invalid choice. Please enter 1 or 2.")
    elif choice == "2":
        print("Choose a dataset :")
        print("1. mauna_loa_atmospheric_co2")
        print("2. international_airline_passengers")
        choice = input("Enter the number of your choice: ")
        if choice == "1":
            fit_pgr_train_bnn_mauna_loa_atmospheric_co2()
        elif choice == "2":
            fit_pgr_train_bnn_international_airline_passengers()
        else:
            print("Invalid choice. Please enter 1 or 2.")
    else:
        print("Invalid choice. Please enter 1 or 2.")


def banchmark(lm, ld):
    # Load evaluation data

    data = []
    for i in range(len(lm)):
        y, X = ld[0]
        X1_train_tensor = torch.from_numpy(X).float()
        y1_train_tensor = torch.from_numpy(y).float()
        model = torch.load(lm[0])
        with torch.no_grad():
            model.eval()
            predictions_1 = model(X1_train_tensor)
            loss_function = nn.MSELoss()
            mse = loss_function(predictions_1, y1_train_tensor)
        data.append([lm[i].split("/")[-1].split(".")[0], X.shape[0], mse])
    # print the table .
    headers = ["Model", "data size ", "Mse "]
    print(tabulate(data, headers, tablefmt="grid"))


lm = [
    "./models/mauna_loa_model.pth",
    "./models/international_airline_passengers_model.pth",
]

ld = [
    load_mauna_loa_atmospheric_co2("data/mauna_loa_atmospheric_co2.csv")[1:],
    load_international_airline_passengers("data/international-airline-passengers.csv")[
        1:
    ],
]

test()
# banchmark(lm, ld)
