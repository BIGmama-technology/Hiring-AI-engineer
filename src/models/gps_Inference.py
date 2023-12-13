import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from models.GaussianProcess import GaussianProcess
from models.kernels import (
    GaussianKernel,
    RBFKernel,
    RationalQuadraticKernel,
    ExponentiatedKernelSineKernel
)
from data.data_loader import (
    load_data_set
)

# Define the kernels that we'll be using
kernels = [GaussianKernel(), RBFKernel(), RationalQuadraticKernel(), ExponentiatedKernelSineKernel()]
kernel_names = ['Gaussian', 'RBF', 'Rational Quadratic', 'Exponentiated Sine']

# defining data files paths
MUANA_DATA_PATH = "data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "data/international-airline-passengers.csv"

# ------------------------------------------
# mauna_loa_atmospheric_co2 Dataset
# ------------------------------------------

# Load the data using a generic function
features, targets, features_normalized = load_data_set(MUANA_DATA_PATH, "decimal date", "average")

# Split the data into training and test sets
features_train, features_test, targets_train, targets_test = train_test_split(
    features_normalized, targets, test_size=0.2, random_state=42, shuffle=False
)

# Initialize plot
plt.figure(figsize=(20, 10))
plt.title("Mean Predictions with Confidence Intervals (95%)")
plt.xticks([])
plt.yticks([])

# optimazed with a loop over each kernel
for i, kernel in enumerate(kernels):
    # Train the model
    gp = GaussianProcess(kernel)
    gp.fit(features_train, targets_train)

    # Predict the mean and covariance
    mean, cov = gp.predict(features_test)

    # Calculate 95% confidence intervals
    # where upper is the upper bound of interval
    # and lower is the lower bound of interval
    std = np.sqrt(np.diag(cov))
    upper = mean + 2 * std
    lower = mean - 2 * std

    # Plot the results
    ########################
    #creating subplot grid to be able to compare the results
    plt.subplot(2, 2, i+1)

    # plot of all data points
    plt.scatter(features_normalized, targets, color="black", s=1)

    # plot of the mean result
    plt.plot(features_test, mean, label=f"'mean' {kernel_names[i]} Kernel")

    # plot the confidence interval
    plt.fill_between(features_test.flatten(), upper, lower, alpha=0.5, label=f"{kernel_names[i]} Kernel")

    # plot informations
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()


plt.show()

# ------------------------------------------
# international-airline-passengers Dataset
# ------------------------------------------

# Load the data using a generic function
features, targets, features_normalized = load_data_set(AIRLINE_DATA_PATH, "Month", "Passengers", convert_to_int=True)

# Split the data into training and test sets
features_train, features_test, targets_train, targets_test = train_test_split(
    features_normalized, targets, test_size=0.2, random_state=42, shuffle=False
)

# Initialize plot
plt.figure(figsize=(15, 10))
plt.title("Mean Predictions with Confidence Intervals (95%)")
plt.xticks([])
plt.yticks([])

# optimazed with a loop over each kernel
for i, kernel in enumerate(kernels):
    # Train the model
    gp = GaussianProcess(kernel)
    gp.fit(features_train, targets_train)

    # Predict the mean and covariance
    mean, cov = gp.predict(features_test)

    # Calculate 95% confidence intervals
    std = np.sqrt(np.diag(cov))
    upper = mean + 2 * std
    lower = mean - 2 * std

    # Plot the results
    ########################
    #creating subplot grid to be able to compare the results
    plt.subplot(2, 2, i+1)

    # plot of all data points
    plt.scatter(features_normalized, targets, color="black", s=1)

    # plot of the mean result
    plt.plot(features_test, mean, label=f"'mean' {kernel_names[i]} Kernel")

    # plot the confidence interval
    plt.fill_between(features_test.flatten(), upper, lower, alpha=0.5, label=f"{kernel_names[i]} Kernel")

    # plot informations
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

plt.show()