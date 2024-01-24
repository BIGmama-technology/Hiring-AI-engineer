import torch
import numpy as np
import matplotlib.pyplot as plt
from MLPModel import MLP
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load the trained model
model_file = 'mlp_time_series_model.pth'
input_dim = 3  # same as look_back in training
hidden_dim = 50
output_dim = 1
dropout_rate = 0.5
model = MLP(input_dim, hidden_dim, output_dim, dropout_rate)
model.load_state_dict(torch.load(model_file))
model.eval()

# load and preprocess the dataset for inference
file_path = 'data/international-airline-passengers.csv'
data = pd.read_csv(file_path)
data['Month'] = pd.to_datetime(data['Month'])
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Passengers'].values.reshape(-1, 1))

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 3
X, y = create_dataset(data_scaled, look_back)
X_test = np.reshape(X, (X.shape[0], X.shape[1], 1))
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Monte Carlo Dropout for uncertainty estimation
def predict_with_uncertainty(model, input_tensor, n_iter=100):
    model.train()  # Enable dropout
    preds = [model(input_tensor).detach().numpy() for _ in range(n_iter)]
    preds = np.array(preds)
    mean_preds = preds.mean(axis=0)
    uncertainty = preds.std(axis=0)
    return mean_preds, uncertainty

# predict on the data
mean_preds, uncertainty = predict_with_uncertainty(model, X_test_tensor)

# inverse transform the predictions and uncertainty
mean_preds_rescaled = scaler.inverse_transform(mean_preds)
uncertainty_rescaled = scaler.inverse_transform(uncertainty)

# plotting
plt.fill_between(range(len(mean_preds_rescaled)), 
                 (mean_preds_rescaled - uncertainty_rescaled).squeeze(), 
                 (mean_preds_rescaled + uncertainty_rescaled).squeeze(), color='gray', alpha=0.5)
plt.plot(range(len(mean_preds_rescaled)), mean_preds_rescaled.squeeze(), label='Predicted Mean')
plt.scatter(range(len(data_scaled)), data['Passengers'], color='red', label='Actual Data')
plt.title('MLP Predictions with Uncertainty')
plt.xlabel('Time')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()
