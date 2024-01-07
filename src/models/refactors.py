import os 
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.distributions import Normal
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error



class Refactor_var():
    """
    A class for training, evaluating, and visualizing a neural network model.

    Parameters:
        - train_data (tuple): Tuple (or list) with len = 2 containing training input and output data.
        - test_data (tuple) : Tuple (or list) with len = 2 containing test input and output data.
        - model (torch.nn.Module): Neural network model to be trained and evaluated.
        - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters during training.
        - epochs (int): Number of training epochs.

    Attributes:
        - train_inp (torch.Tensor): Training input data.
        - train_out (torch.Tensor): Training output data.
        - test_inp (torch.Tensor) : Test input data.
        - test_out (torch.Tensor) : Test output data.
        - net (torch.nn.Module)   : Neural network model.
        - optimizer (torch.optim.Optimizer): Optimizer.
        - epochs (int): Number of training epochs.
        - train_losses (list): List to store training losses.
        - predictions (numpy.ndarray or None): Array to store predictions or None if not evaluated.
    """
    def __init__(self, train_data, test_data, model, optimizer, epochs):
       # Check if train_data and test_data are tuples of length 2
        assert len(train_data) == 2, "train_data must be of length 2"
        assert len(test_data) == 2, "test_data must be of length 2"

        # Assign attributes
        self.train_inp, self.train_out = train_data
        self.test_inp, self.test_out = test_data

        # Check if model is a torch.nn.Module
        assert isinstance(model, torch.nn.Module), "model must be an instance of torch.nn.Module"
        self.net = model

        # Check if optimizer is a torch.optim.Optimizer
        assert isinstance(optimizer, torch.optim.Optimizer), "optimizer must be an instance of torch.optim.Optimizer"
        self.optimizer = optimizer

        # Check if epochs is an integer
        assert isinstance(epochs, int), "epochs must be an integer"
        self.epochs = epochs

        # List to store training losses
        self.train_losses = []

        # Variable to store predictions
        self.predictions = None


    def fit(self):
        """
            Train the neural network model.

            Updates the model parameters based on the training data.

            Returns: None
        """
        # Create a tqdm progress bar for epochs
        progress_bar_epochs = tqdm(total=self.epochs, desc='Training the model: ', unit='epoch')
        
        # Clear the list of training losses
        self.train_losses = []

        for epoch in range(self.epochs):
            # Forward pass
            outputs = self.net(self.train_inp)
            loss = self.elbo_loss(self.train_inp, self.train_out)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Append the loss to the list
            self.train_losses.append(loss.item())

            # Update the progress bar for epochs
            progress_bar_epochs.update(1)

        # Close the progress bar for epochs
        progress_bar_epochs.close()

        # Display a message after training is done
        print("Training done!\n\n")

    def elbo_loss(self, input, target, reps = 7):
        """
        Calculate the Evidence Lower Bound (ELBO) loss for the given input and target.

        Parameters:
            - input (torch.Tensor): Input data for the neural network.
            - target (torch.Tensor): Target data for the neural network.

        Returns:
            torch.Tensor: ELBO loss value.
        """
        # Set the number of Monte Carlo samples (reps)
        

        # Compute the Kullback-Leibler (KL) divergence term
        kl = self.net.kl_net() / input.shape[0]

        # Initialize the variable to accumulate predictions
        pred = 0

        # Monte Carlo estimation of the predicted values
        for _ in range(reps): 
            pred += self.net.forward(input) / reps

        # Calculate the log probability of the target given the predicted values
        log_p = Normal(pred, 1).log_prob(target).sum()

        # Calculate and return the negative ELBO loss, so it's supposed to go up in order to get kl lower 
        return - (log_p - kl)

    def eval(self, plot=True):
        
        """
            Evaluate the neural network model on the test data.
    
            Parameters:
            - plot (bool): If True, plot Ground Truth vs Predictions.
    
            Returns: 
                prediction values if plot parameter is false, None else
            
        """
        with torch.no_grad():
            # Set the model to evaluation mode
            self.net.eval()
            # Make predictions on the test data
            predictions_1 = self.net(self.test_inp)
            # Set the model back to training mode
            self.net.train()

        # Convert predictions to NumPy array for plotting
        predictions_np_1 = predictions_1.numpy()

        if plot:
            # Calculate evaluation metrics
            r2 = r2_score(predictions_np_1, self.test_out)
            mse = mean_squared_error(predictions_np_1, self.test_out)
            mae = mean_absolute_error(predictions_np_1, self.test_out)

            # Plot Ground Truth vs Predictions
            plt.figure(figsize=(10, 6))
            plt.plot(self.test_inp, self.test_out, "b.", markersize=10, label="Ground Truth")
            plt.plot(self.test_inp, predictions_1, "r.", markersize=10, label="Predictions")
            plt.xlabel("Input Features (X_test)")
            plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
            plt.title("True Values vs Predictions")
            plt.legend()

            # Add the MSE, R2, and MAE values to plot
            plt.text(0.8, 0.1, f'MSE: {mse:.4f}\nR2: {r2:.4f}\nMAE: {mae:.4f}',
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), weight='bold')

            plt.show()
        else:
            # Save predictions for later use
            self.predictions = predictions_np_1
            return self.predictions

    def plot_loss(self):
        """
            Plot the training losses over epochs.
    
            Returns:
            None
        """
        # Plot training losses over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs + 1), self.train_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.show()

    def save_model(self, path, name):
        """
        Save the trained model to the specified path with the specified name.

        Parameters:
        - path (str): The directory path to save the model.
        - name (str): The name to use for saving the model.

        Returns:
        None
        """
        # Check if the specified path exists, create it if it doesn't
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Creating it.")
            os.makedirs(path)

        # Ensure the name ends with '.pth'
        if not name.endswith('.pth'):
            name += '.pth'

        # Construct the full file path
        file_path = os.path.join(path, name)

        # Save the trained model to the specified path and name
        torch.save(self.net, file_path)

        print(f"Model saved to {file_path}")

class Refactor():
    """
    A class for training, evaluating, and visualizing a neural network model.

    Parameters:
        - train_data (tuple): Tuple (or list) with len = 2 containing training input and output data.
        - test_data (tuple) : Tuple (or list) with len = 2 containing test input and output data.
        - model (torch.nn.Module): Neural network model to be trained and evaluated.
        - criterion : Loss criterion for training if the model isn't based on variational inference and ELBO loss function.
        - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters during training.
        - epochs (int): Number of training epochs.

    Attributes:
        - train_inp (torch.Tensor): Training input data.
        - train_out (torch.Tensor): Training output data.
        - test_inp (torch.Tensor) : Test input data.
        - test_out (torch.Tensor) : Test output data.
        - net (torch.nn.Module)   : Neural network model.
        - criterion (torch.nn.Module): Loss criterion.
        - optimizer (torch.optim.Optimizer): Optimizer.
        - epochs (int): Number of training epochs.
        - train_losses (list): List to store training losses.
        - predictions (numpy.ndarray or None): Array to store predictions or None if not evaluated.
    """
    def __init__(self, train_data, test_data, model, criterion, optimizer, epochs):
       # Check if train_data and test_data are tuples of length 2
        assert len(train_data) == 2, "train_data must be of length 2"
        assert len(test_data) == 2, "test_data must be of length 2"

        # Assign attributes
        self.train_inp, self.train_out = train_data
        self.test_inp, self.test_out = test_data

        # Check if model is a torch.nn.Module
        assert isinstance(model, torch.nn.Module), "model must be an instance of torch.nn.Module"
        self.net = model

        self.criterion = criterion

        # Check if optimizer is a torch.optim.Optimizer
        assert isinstance(optimizer, torch.optim.Optimizer), "optimizer must be an instance of torch.optim.Optimizer"
        self.optimizer = optimizer

        # Check if epochs is an integer
        assert isinstance(epochs, int), "epochs must be an integer"
        self.epochs = epochs

        # List to store training losses
        self.train_losses = []

        # Variable to store predictions
        self.predictions = None


    def fit(self):
        """
            Train the neural network model.

            Updates the model parameters based on the training data.

            Returns: None
        """
        # Create a tqdm progress bar for epochs
        progress_bar_epochs = tqdm(total=self.epochs, desc='Training the model: ', unit='epoch')
        
        # Clear the list of training losses
        self.train_losses = []

        for epoch in range(self.epochs):
            # Forward pass
            outputs = self.net(self.train_inp)
            loss = self.criterion(outputs, self.train_out)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Append the loss to the list
            self.train_losses.append(loss.item())

            # Update the progress bar for epochs
            progress_bar_epochs.update(1)

        # Close the progress bar for epochs
        progress_bar_epochs.close()

        # Display a message after training is done
        print("Training done!\n\n")

    def eval(self, plot=True):
        
        """
            Evaluate the neural network model on the test data.
    
            Parameters:
            - plot (bool): If True, plot Ground Truth vs Predictions.
    
            Returns: 
                prediction values if plot parameter is false, None else
            
        """
        with torch.no_grad():
            # Set the model to evaluation mode
            self.net.eval()
            # Make predictions on the test data
            predictions_1 = self.net(self.test_inp)
            # Set the model back to training mode
            self.net.train()

        # Convert predictions to NumPy array for plotting
        predictions_np_1 = predictions_1.numpy()

        if plot:
            # Calculate evaluation metrics
            r2 = r2_score(predictions_np_1, self.test_out)
            mse = mean_squared_error(predictions_np_1, self.test_out)
            mae = mean_absolute_error(predictions_np_1, self.test_out)

            # Plot Ground Truth vs Predictions
            plt.figure(figsize=(10, 6))
            plt.plot(self.test_inp, self.test_out, "b.", markersize=10, label="Ground Truth")
            plt.plot(self.test_inp, predictions_1, "r.", markersize=10, label="Predictions")
            plt.xlabel("Input Features (X_test)")
            plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
            plt.title("True Values vs Predictions")
            plt.legend()

            # Add the MSE, R2, and MAE values to plot
            plt.text(0.8, 0.1, f'MSE: {mse:.4f}\nR2: {r2:.4f}\nMAE: {mae:.4f}',
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), weight='bold')

            plt.show()
        else:
            # Save predictions for later use
            self.predictions = predictions_np_1
            return self.predictions

    def plot_loss(self):
        """
            Plot the training losses over epochs.
    
            Returns:
            None
        """
        # Plot training losses over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs + 1), self.train_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.show()

    def save_model(self, path, name):
        """
        Save the trained model to the specified path with the specified name.

        Parameters:
        - path (str): The directory path to save the model.
        - name (str): The name to use for saving the model.

        Returns:
        None
        """
        # Check if the specified path exists, create it if it doesn't
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Creating it.")
            os.makedirs(path)

        # Ensure the name ends with '.pth'
        if not name.endswith('.pth'):
            name += '.pth'

        # Construct the full file path
        file_path = os.path.join(path, name)

        # Save the trained model to the specified path and name
        torch.save(self.net, file_path)

        print(f"Model saved to {file_path}")
        
        