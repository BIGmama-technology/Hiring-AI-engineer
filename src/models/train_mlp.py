import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from MLPModel import MLP
import os

# load and preprocess the dataset
file_path = 'data/international-airline-passengers.csv'
data = pd.read_csv(file_path)

# convert 'Month' column to datetime
data['Month'] = pd.to_datetime(data['Month'])

# normalize 'Passengers' data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Passengers'].values.reshape(-1, 1))

# create dataset for time series forecasting
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# look_back period : number of previous time steps to consider for predicting the next value
look_back = 3
X, y = create_dataset(data_scaled, look_back)

# split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# convert data to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# cefine the model, loss function and optimizer
input_dim = look_back
hidden_dim = 50
output_dim = 1
dropout_rate = 0.5
model = MLP(input_dim, hidden_dim, output_dim, dropout_rate)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# save the trained model
model_file = 'mlp_time_series_model.pth'
torch.save(model.state_dict(), model_file)
print(f"Model saved to {model_file}")
