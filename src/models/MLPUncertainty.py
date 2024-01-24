import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from MLPModel import MLP

# preprocess the dataset before training
# reading csv
file_path = 'data\international-airline-passengers.csv' 
data = pd.read_csv(file_path)

# convert 'Month' column to datetime
data['Month'] = pd.to_datetime(data['Month'])

# sort the data by month
data.sort_values('Month', inplace=True)

# normalize 'Passengers' data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Passengers'].values.reshape(-1,1))

# create dataset for time series forecasting
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# look_back period : number of previous time steps to consider for predicting the next value
look_back = 1 # month
X, y = create_dataset(scaled_data, look_back)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# initialize the model
input_size = 1  # we have 1 feature per time step
hidden_size = 50
output_size = 1  # we prredict a single value
dropout_rate = 0.5  # dropout rate

model = MLP(input_size, hidden_size, output_size, dropout_rate)

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# convert data to torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

# training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_torch)
    loss = criterion(output, y_train_torch)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# in the below portion of code, we use the trained mlp for predictions and visualize uncertainty.

def predict_with_uncertainty(model, input_data, n_iter=100):
    predictions = []
    for _ in range(n_iter):
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = model(input_data)
        predictions.append(output.numpy())
    
    predictions = np.array(predictions)
    mean_predictions = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    return mean_predictions, uncertainty

# making predictions with uncertainty
mean_predictions, uncertainty = predict_with_uncertainty(model, X_test_torch)

# inverse transform the predictions and uncertainty to original scale
mean_predictions_inv = scaler.inverse_transform(mean_predictions.reshape(-1, 1)).flatten()
uncertainty_inv = scaler.inverse_transform(uncertainty.reshape(-1, 1)).flatten()

# plotting
plt.figure(figsize=(12, 6))
plt.plot(data['Month'][-len(y_test):], mean_predictions_inv, label='Predictions')
plt.fill_between(data['Month'][-len(y_test):], 
                 (mean_predictions_inv - uncertainty_inv), 
                 (mean_predictions_inv + uncertainty_inv), 
                 color='gray', alpha=0.2, label='Uncertainty')
plt.scatter(data['Month'][-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='red', label='Actual')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.title('Predictions with Uncertainty')
plt.legend()
plt.show()




