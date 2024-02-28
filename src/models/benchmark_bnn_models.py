import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.BnnModel import BayesianModel
from src.data.data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)

MUANA_DATA_PATH = "./data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "./data/international-airline-passengers.csv"

# Load test data for mauna_loa_atmospheric_co2 dataset
X1, y1, X1_normalized = load_mauna_loa_atmospheric_co2(MUANA_DATA_PATH)
X1_test_tensor = torch.from_numpy(X1_normalized).float()
y1_test = torch.from_numpy(y1).float()

# Load test data for international-airline-passengers dataset
X2, y2, X2_normalized = load_international_airline_passengers(AIRLINE_DATA_PATH)
X2_test_tensor = torch.from_numpy(X2_normalized).float()
y2_test = torch.from_numpy(y2).float()


def evaluate_model(model, test_data, true_labels):
    model.eval()
    with torch.no_grad():
        predictions = model.predict(test_data)

    mse = mean_squared_error(true_labels, predictions.numpy())
    mae = mean_absolute_error(true_labels, predictions.numpy())
    rmse = mean_squared_error(true_labels, predictions.numpy(), squared=False)
    r2 = r2_score(true_labels, predictions.numpy())

    return mse, mae, rmse, r2


def main():
    # Load trained models
    model_mauna = torch.load("./models/bnn_mauna.pth")
    model_airline = torch.load("./models/bnn_airline.pth")

    # Evaluate models on mauna_loa_atmospheric_co2 dataset
    mse_mauna, mae_mauna, rmse_mauna, r2_mauna = evaluate_model(
        model_mauna, X1_test_tensor, y1_test
    )

    print("Metrics for mauna_loa_atmospheric_co2 dataset:")
    print(f"MSE: {mse_mauna}")
    print(f"MAE: {mae_mauna}")
    print(f"RMSE: {rmse_mauna}")
    print(f"R-squared: {r2_mauna}")
    print("\n")

    # Evaluate models on international-airline-passengers dataset
    mse_airline, mae_airline, rmse_airline, r2_airline = evaluate_model(
        model_airline, X2_test_tensor, y2_test
    )

    print("Metrics for international-airline-passengers dataset:")
    print(f"MSE: {mse_airline}")
    print(f"MAE: {mae_airline}")
    print(f"RMSE: {rmse_airline}")
    print(f"R-squared: {r2_airline}")


if __name__ == "__main__":
    main()
