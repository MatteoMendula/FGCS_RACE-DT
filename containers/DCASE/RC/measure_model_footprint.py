from sklearn.neural_network import MLPRegressor
import joblib
import os

# --- MLP Readout Configuration ---
MLP_HIDDEN_LAYER_SIZES = (64, 32) # Example: two hidden layers, can be tuned or fixed
MLP_ACTIVATION = 'relu'
MLP_SOLVER = 'adam'
MLP_MAX_ITER = 300 # Increased for potentially more complex task
MLP_EARLY_STOPPING = True
MLP_N_ITER_NO_CHANGE = 10
MLP_ALPHA = 0.0001 # L2 regularization
reservoir_seed = 42 # Example seed for reproducibility

# Instantiate the MLPRegressor model
mlp_readout = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                           activation=MLP_ACTIVATION,
                           solver=MLP_SOLVER,
                           max_iter=MLP_MAX_ITER,
                           early_stopping=MLP_EARLY_STOPPING,
                           n_iter_no_change=MLP_N_ITER_NO_CHANGE,
                           random_state=reservoir_seed, # For reproducibility
                           alpha=MLP_ALPHA)

# Define the filename for the saved model
model_filename = 'untrained_mlp_regressor.joblib'

# Save the model to a file
joblib.dump(mlp_readout, model_filename)

print(f"Model saved to {model_filename}")

# To measure the footprint, you can check the size of the saved file
file_size = os.path.getsize(model_filename)
print(f"The footprint of the saved model is: {file_size} bytes")

# You can also load the model back to verify (optional)
# loaded_model = joblib.load(model_filename)
# print("Model loaded successfully (optional check)")