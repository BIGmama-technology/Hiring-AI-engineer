import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
from models.train_bnn import *


def run_bnn():
    model = BayesianModel(input_size, hidden_size, output_size, kl_weight, torch.relu)

    # Define loss function and optimizer
    loss_function = neg_log_likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    for t in tqdm.tqdm(range(num_epochs)):
        train_losses += train_loop(train_dataloader, model, loss_function, optimizer)

    print("Done!")
    # ---------  Plot training losses  ----------

    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

    # --------- Plot epistemic uncertainty  ------------
    y_pred_list = []
    X_test = torch.tensor(np.linspace(-4, 4, 1000).reshape(-1, 1)).float()
    for i in tqdm.tqdm(range(num_epochs)):
        y_pred = model(X_test)
        y_pred_list.append(y_pred.detach().numpy())

    y_preds = np.concatenate(y_pred_list, axis=1)

    y_mean = np.mean(y_preds, axis=1)
    y_sigma = np.std(y_preds, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_mean, "r-", label="Predictive mean")
    plt.scatter(X1_test, y1_test, marker="+", label="Training data")
    plt.fill_between(
        X_test.ravel(),
        y_mean + 2 * y_sigma,
        y_mean - 2 * y_sigma,
        alpha=0.5,
        label="Epistemic uncertainty",
    )
    plt.xlabel("Input Features (X_test)")
    plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
    plt.title("True Values vs Predictions")
    plt.legend()
    plt.show()

    # export model
    torch.save(model, "./models/mauna_loa_model.pth")


# run bnn_train program
run_bnn()