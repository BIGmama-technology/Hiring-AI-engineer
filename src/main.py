# eliasboulham@gmail.com
from src.models.BnnModel import *
from src.models.train_bnn import *
from src.models.BnnModel import *
from src.models.Fit_gaussian_process import *

import torch
import numpy as np
import os
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


test()
