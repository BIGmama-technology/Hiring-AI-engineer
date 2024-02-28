# Benchmarking Model

## Overview

Follow the steps below to benchmark the model's performance on relevant metrics.

## Benchmarking the Example BNN Model

1. **Prepare the Dataset:**
   - Ensure you have a suitable dataset for benchmarking the BNN model. The required datasets (`mauna_loa_atmospheric_co2.csv` and `international-airline-passengers.csv`) should be placed in the `./data` directory.

2. **Train the Model:**
   - Train the BNN models using the provided script. The trained models will be saved in the `./models` directory.

   ```bash
   python train_bnn_models.py
3. **Benchmarking:**

- Run the benchmarking script to evaluate the models on the metrics.
    ```bash
    python benchmark_bnn_models.py

## Benchmarking Future Models
To benchmark future models, follow these steps:

- Implement the New Model:
Implement the new model in the `src.models` directory.


- Update the Training Script:
Update the `train_models.py` script to include training for the new model.
Save the New Model:

- Ensure that the new model is saved using torch.save in the `./models` directory.

- Run the Benchmarking Script:
Run the benchmarking script (`benchmark_models.py`) to evaluate the new model alongside existing models. Adjust the script as needed.


