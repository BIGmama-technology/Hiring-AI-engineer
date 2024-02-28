## Benchmarking the BNN Model

To benchmark a model, follow these steps:

1. Prepare the dataset:
Ensure that you have a suitable dataset for benchmarking the model Example BNN model. Ensure that you have the required datasets (mauna_loa_atmospheric_co2.csv and international-airline-passengers.csv) in the ./data directory.

2. Train the model:
Train the BNN models using the provided script. The trained models will be saved in the ./models directory.
`python train_bnn_models.py`

3. Benchmarking:
Run the benchmarking script to evaluate the models on relevant metrics.
`python benchmark_bnn_models.py`



To benchmark future models, follow these steps:

1. Implement the new model in the src.models directory.

2. Update the train_models.py script to include training for the new model.

3. Ensure that the new model is saved using torch.save in the ./models directory.

4. Run the benchmarking script (benchmark_models.py) to evaluate the new model alongside existing models.

