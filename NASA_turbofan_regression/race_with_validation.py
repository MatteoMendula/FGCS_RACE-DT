import os
import numpy as np
# import librosa # Removed
from reservoirpy.nodes import Reservoir, IPReservoir
from reservoirpy.mat_gen import uniform, bernoulli
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split # Added for validation split
# import random # No longer explicitly used here, but good for general reproducibility if seed is set
# from collections import deque # No longer explicitly used
# import copy # No longer explicitly used
import traceback
from reservoirpy.utils import verbosity
# from myRLS import RLS # RLS is no longer used
# Assuming this base class is available
from baseEpsilonGreedyMultiReservoirHPSearch import BaseEpsilonGreedyMultiReservoirHPSearch # Assuming this exists
import sys
import time
import pandas as pd
import json

verbosity(0)

# === PREDICTION SCRIPT CONFIGS FOR FD01 DATA ===
# SR = 16000 # Removed
# N_MFCC = 13 # Removed, will be derived from data

def count_sklearn_mlp_macs_and_params(mlp_model):
    """
    Calculates the total MACs and parameters for a scikit-learn MLPRegressor or MLPClassifier. 
    """
    total_macs = 0
    if not hasattr(mlp_model, 'coefs_') or not hasattr(mlp_model, 'intercepts_'):
        print("MLP model is not fitted or does not have coefficients/intercepts.")
        return 0

    # MACs = sum of (inputs_to_layer_i * outputs_of_layer_i) for all layers
    # For a dense layer, this is shape[0] * shape[1] of the weight matrix
    for coef_matrix in mlp_model.coefs_:
        # coef_matrix.shape = (n_inputs_to_this_layer, n_outputs_from_this_layer)
        macs_layer = coef_matrix.shape[0] * coef_matrix.shape[1]
        total_macs += macs_layer

    n_params_mlp = sum(c.size for c in mlp_model.coefs_) + sum(i.size for i in mlp_model.intercepts_) if hasattr(mlp_model, 'coefs_') else 0
    return total_macs, n_params_mlp

def getResultsFolder(asset_name):
    RESULTS_FOLDER = f"./NASA_results_{asset_name}/race_free_mlp_aggregated_prediction_val_split/" # Changed folder name
    return RESULTS_FOLDER

# --- .npy File Paths ---
# Assumes these files are in the same directory as the script, or provide full paths.
def get_data_file_path(asset_name):
    DATA_FILE_PATHS = {
        "x_train": f'./old_data/x_train_{asset_name}.npy',
        "y_train": f'./old_data/y_train_{asset_name}.npy',
        "x_test": f'./old_data/x_test_{asset_name}.npy',
        "y_test": f'./old_data/y_test_{asset_name}.npy'
    }
    return DATA_FILE_PATHS

# --- MLP Readout Configuration ---
MLP_HIDDEN_LAYER_SIZES = (64, 32)
MLP_ACTIVATION = 'relu'
MLP_SOLVER = 'adam'
MLP_MAX_ITER = 300
MLP_EARLY_STOPPING = True
MLP_N_ITER_NO_CHANGE = 10
MLP_ALPHA = 0.0001

# --- HYPERPARAMETER SEARCH CONFIGURATION ---
SEARCH_SPACE = {
    "units": [5, 50], # Expanded example, tune further
    "sr": {"min": np.log(0.1), "max": np.log(1.5)},
    "lr": [0.1, 0.5, 1.0], # Expanded example
    "input_scaling": {"min": np.log(0.05), "max": np.log(2.0)},
    "connectivity": [0.01, 0.1, 0.5], # Expanded example
    "activation": ["tanh", "sigmoid"],
    "RC_node_type": ["ESN", "IPESN"],
    "mu": [0.0, 0.1, 0.2], # For IPESN
    "learning_rate": {"min": np.log(1e-5), "max": np.log(1e-2)}, # For IPESN
}

GLOBAL_PARAMS = {
    "seed": 1234,
    "warmup": 0, # Consider 0 if sequences are always length 1, otherwise tune
    "epochs": 100, # For IPESN
}
N_HP_ITERATIONS = 5 # Per layer (increased for better search with validation)
VALIDATION_SET_SIZE = 0.4 # Proportion of training data to use for validation during HP search

def save_csv_results(results, file_full_path):
    os.makedirs(os.path.dirname(file_full_path), exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(file_full_path, index=False)
    print(f"Results saved to {file_full_path}")

def load_FD_data_as_sequences(file_paths):
    try:
        X_train_np = np.load(file_paths['x_train'])
        y_train_np = np.load(file_paths['y_train'])
        X_test_np = np.load(file_paths['x_test'])
        y_test_np = np.load(file_paths['y_test'])
    except FileNotFoundError as e:
        print(f"Error loading .npy files: {e}. Ensure all files are present: {file_paths}")
        return None, None, None, None

    print(f"Loaded X_train shape: {X_train_np.shape}, y_train shape: {y_train_np.shape}")
    print(f"Loaded X_test shape: {X_test_np.shape}, y_test shape: {y_test_np.shape}")

    if X_train_np.ndim == 2:
        X_list_train_raw = [X_train_np[i:i+1, :] for i in range(X_train_np.shape[0])]
    elif X_train_np.ndim == 3:
        X_list_train_raw = [X_train_np[i, :, :] for i in range(X_train_np.shape[0])]
    else:
        raise ValueError("X_train_np has unsupported dimension. Expected 2D or 3D.")

    if X_test_np.ndim == 2:
        X_list_test_raw = [X_test_np[i:i+1, :] for i in range(X_test_np.shape[0])]
    elif X_test_np.ndim == 3:
        X_list_test_raw = [X_test_np[i, :, :] for i in range(X_test_np.shape[0])]
    else:
        raise ValueError("X_test_np has unsupported dimension. Expected 2D or 3D.")

    y_train_targets_raw = y_train_np.ravel()
    y_test_targets_raw = y_test_np.ravel()

    if len(X_list_train_raw) != len(y_train_targets_raw):
        raise ValueError("Mismatch in number of training samples and training targets.")
    if len(X_list_test_raw) != len(y_test_targets_raw):
        raise ValueError("Mismatch in number of test samples and test targets.")

    return X_list_train_raw, y_train_targets_raw, X_list_test_raw, y_test_targets_raw


class PredictionHPSearch(BaseEpsilonGreedyMultiReservoirHPSearch):
    def __init__(self, input_dim, X_hp_train_data, y_hp_train_data, # Renamed for clarity
                 X_hp_val_data, y_hp_val_data,       # New validation set parameters
                 n_iterations, n_reservoirs, reservoir_hp_space, global_params,
                 epsilon_greedy=0.3):

        # The base class `BaseEpsilonGreedyMultiReservoirHPSearch` is assumed to store
        # its first four data arguments as self.X_train, self.y_train, self.X_test, self.y_test.
        # For the purpose of HP search, we pass:
        # - X_hp_train_data as X_train to the base
        # - y_hp_train_data as y_train to the base
        # - X_hp_val_data as X_test to the base (this will be our validation set)
        # - y_hp_val_data as y_test to the base (this will be our validation set targets)
        super().__init__(input_dim, X_hp_train_data, y_hp_train_data,
                         X_hp_val_data, y_hp_val_data, # These act as validation set for HP search
                         n_iterations, n_reservoirs, reservoir_hp_space, global_params, epsilon_greedy)
        
        # No need to store X_hp_train_data etc. separately if super() handles it as self.X_train etc.
        # We just need to remember that within this class's evaluate method:
        # self.X_train refers to the HP search's training portion.
        # self.y_train refers to the HP search's training targets.
        # self.X_test refers to the HP search's validation portion.
        # self.y_test refers to the HP search's validation targets.

    def _get_aggregated_reservoir_outputs_and_targets(self, X_data, Y_targets_data, reservoirs, warmup_glob):
        aggregated_inputs = []
        aggregated_targets = []

        if not X_data or len(X_data) != len(Y_targets_data): # check if X_data is empty
            print("Error: Mismatch between number of input sequences and target values, or X_data is empty.")
            return None, None

        for i, x_seq_single in enumerate(X_data):
            y_target_for_sequence = Y_targets_data[i]

            for res_node in reservoirs:
                res_node.reset()

            if len(x_seq_single) == 0: continue

            effective_inputs = x_seq_single
            if warmup_glob > 0 and len(x_seq_single) > warmup_glob :
                input_warmup = x_seq_single[:warmup_glob]
                current_layer_warmup = input_warmup
                for res_node in reservoirs:
                    _ = res_node.run(current_layer_warmup, reset=False)
                    current_layer_warmup = _
                    if current_layer_warmup.shape[0] == 0: break
                effective_inputs = x_seq_single[warmup_glob:]
            elif warmup_glob > 0 and len(x_seq_single) <= warmup_glob:
                # If sequence is shorter than warmup, process without explicit warmup cut if warmup > 0.
                # Fallback will handle using the full (short) sequence.
                pass # Process the whole sequence

            if effective_inputs.shape[0] == 0:
                if len(x_seq_single) > 0:
                    current_sequence_data_fb = x_seq_single
                    for res_node in reservoirs:
                        output_states_sequence_fb = res_node.run(current_sequence_data_fb, reset=False)
                        current_sequence_data_fb = output_states_sequence_fb
                        if current_sequence_data_fb.shape[0] == 0: break
                    if current_sequence_data_fb.shape[0] > 0:
                        final_layer_output_sequence = current_sequence_data_fb[-1:,:]
                    else:
                        continue
                else:
                    continue
            else:
                current_sequence_data = effective_inputs
                for res_node in reservoirs:
                    output_states_sequence = res_node.run(current_sequence_data, reset=False)
                    current_sequence_data = output_states_sequence
                    if current_sequence_data.shape[0] == 0: break
                final_layer_output_sequence = current_sequence_data

            if final_layer_output_sequence.shape[0] == 0: continue

            aggregated_reservoir_output_for_sequence = np.mean(final_layer_output_sequence, axis=0)
            aggregated_inputs.append(aggregated_reservoir_output_for_sequence)
            aggregated_targets.append(y_target_for_sequence)

        if not aggregated_inputs:
            return None, None

        return np.array(aggregated_inputs), np.array(aggregated_targets)


    def evaluate(self, current_params_list):
        if len(current_params_list) != self.n_reservoirs:
            print(f"Error: Expected {self.n_reservoirs} HP sets, got {len(current_params_list)}")
            return -np.inf, None, None, None, None, None, None

        try:
            reservoir_seed = self.global_params.get("seed", None)
            warmup_glob = self.global_params.get("warmup", 0)

            reservoirs = []
            current_input_dim_for_layer = self.input_dim

            for i, layer_hp in enumerate(current_params_list):
                reservoir_node_type = layer_hp.get("RC_node_type", "ESN")
                _activation = layer_hp["activation"]
                shared_params = dict(
                    units=layer_hp["units"], input_dim=current_input_dim_for_layer,
                    sr=layer_hp["sr"], lr=layer_hp["lr"],
                    input_scaling=layer_hp["input_scaling"],
                    rc_connectivity=layer_hp.get("connectivity"),
                    activation=_activation, W=uniform(high=1.0, low=-1.0), Win=bernoulli,
                    seed=reservoir_seed + i if reservoir_seed is not None else None
                )
                if reservoir_node_type == "IPESN":
                    node = IPReservoir(**shared_params,
                                      mu=layer_hp.get("mu", 0.0),
                                      learning_rate=layer_hp.get("learning_rate", 1e-3),
                                      epochs=self.global_params.get("epochs", 1))
                else: # ESN
                    node = Reservoir(**shared_params)
                reservoirs.append(node)
                current_input_dim_for_layer = layer_hp["units"]

            # Train MLP on the HP-search's training portion (self.X_train, self.y_train from base class)
            mlp_train_inputs_np, mlp_train_targets_np = self._get_aggregated_reservoir_outputs_and_targets(
                self.X_train, self.y_train, reservoirs, warmup_glob # self.X_train is X_hp_train_data
            )

            if mlp_train_inputs_np is None or mlp_train_inputs_np.shape[0] < 2 : # Need at least 2 samples for some MLP fitters
                # print("Warning: Not enough data generated from HP-train set for MLP training.")
                return -np.inf, None, None, None, None, None, None

            if mlp_train_targets_np.ndim > 1 and mlp_train_targets_np.shape[1] == 1:
                mlp_train_targets_np = mlp_train_targets_np.ravel()

            mlp_readout = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                                       activation=MLP_ACTIVATION, solver=MLP_SOLVER,
                                       max_iter=MLP_MAX_ITER, early_stopping=MLP_EARLY_STOPPING,
                                       n_iter_no_change=MLP_N_ITER_NO_CHANGE,
                                       random_state=reservoir_seed, alpha=MLP_ALPHA)
            
            start_mlp_training_time = time.time()
            try:
                mlp_readout.fit(mlp_train_inputs_np, mlp_train_targets_np)
            except ValueError as e:
                mlp_readout_no_es = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                                                 activation=MLP_ACTIVATION, solver=MLP_SOLVER,
                                                 max_iter=MLP_MAX_ITER * 10, # Longer training if no early stopping
                                                 random_state=reservoir_seed, alpha=MLP_ALPHA)
                mlp_readout_no_es.fit(mlp_train_inputs_np, mlp_train_targets_np)
                mlp_readout = mlp_readout_no_es
            end_mlp_training_time = time.time()
            mlp_training_duration = end_mlp_training_time - start_mlp_training_time

            # Evaluate on the HP-search's validation portion (self.X_test, self.y_test from base class)
            mlp_val_inputs_np, mlp_val_targets_np = self._get_aggregated_reservoir_outputs_and_targets(
                self.X_test, self.y_test, reservoirs, warmup_glob # self.X_test is X_hp_val_data
            )

            if mlp_val_inputs_np is None or mlp_val_inputs_np.shape[0] == 0:
                # print("Warning: Not enough data generated from HP-validation set for evaluation.")
                return -np.inf, None, None, None, None, None, None # Cannot score

            if mlp_val_targets_np.ndim > 1 and mlp_val_targets_np.shape[1] == 1:
                mlp_val_targets_np = mlp_val_targets_np.ravel()
            
            if len(mlp_val_targets_np) == 0: # Check if targets are empty after potential processing
                # print("Warning: HP-validation targets are empty after processing.")
                return -np.inf, None, None, None, None, None, None

            predicted_mlp_outputs_val = mlp_readout.predict(mlp_val_inputs_np)
            
            if len(predicted_mlp_outputs_val) != len(mlp_val_targets_np):
                # print("Warning: Mismatch in length between validation predictions and targets.")
                return -np.inf, None, None, None, None, None, None


            val_mse = mean_squared_error(mlp_val_targets_np, predicted_mlp_outputs_val)
            score = -val_mse # Score is negative MSE on the validation set

            model_tuple = (reservoirs, mlp_readout, mlp_training_duration)
            output_dim_of_last_reservoir = reservoirs[-1].output_dim if reservoirs else None

            return float(score), None, model_tuple, output_dim_of_last_reservoir, None, None, None

        except Exception as e:
            print(f"ERROR during MLP (prediction) evaluation with HPs {current_params_list}: {e}")
            traceback.print_exc()
            return -np.inf, None, None, None, None, None, None

# === MAIN SCRIPT LOGIC ===
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Please provide the number of layers and the asset name.")

    n_layers = int(sys.argv[1])
    ASSET_NAME = sys.argv[2]
    print(f"Number of layers: {n_layers}")
    print(f"Using data asset: {ASSET_NAME}")

    X_list_train_raw, y_train_targets_full, X_list_test_raw, y_test_targets = load_FD_data_as_sequences(get_data_file_path(ASSET_NAME))

    if X_list_train_raw is None:
        print("Failed to load data. Exiting.")
        exit()
    if not X_list_train_raw:
        print("Training data is empty. Exiting.")
        exit()
    input_dim = X_list_train_raw[0].shape[1]
    print(f"Determined input_dim: {input_dim}")

    print("\n--- Preparing Scaler & Scaling Training Data (Input Features) ---")
    # Concatenate all sequences in X_list_train_raw for fitting the scaler
    # This handles cases where sequences might have different lengths if data was 3D
    all_train_samples_for_scaler = np.vstack(X_list_train_raw)
    scaler_x = StandardScaler().fit(all_train_samples_for_scaler)
    X_list_train_scaled_full = [scaler_x.transform(x) for x in X_list_train_raw]

    X_list_test_scaled = []
    if X_list_test_raw:
        X_list_test_scaled = [scaler_x.transform(x) for x in X_list_test_raw]
        print(f"Test data scaled. Num sequences: {len(X_list_test_scaled)}")
    else:
        print("No test data loaded.")

    print(f"\n--- Splitting Full Training Data into HP-Search Training and Validation Sets ({VALIDATION_SET_SIZE*100}%) ---")
    if len(X_list_train_scaled_full) != len(y_train_targets_full):
         raise ValueError("Mismatch lengths of X_list_train_scaled_full and y_train_targets_full before split.")

    # Ensure y_train_targets_full is a 1D array for stratification if needed, or just for consistency
    y_train_targets_full = np.array(y_train_targets_full).ravel()

    if len(X_list_train_scaled_full) < 2 : # train_test_split needs at least 2 samples for each array
        print("Error: Not enough samples in the full training data to create a validation split. Exiting.")
        exit()
    
    # Handle case where y_train_targets_full might lead to very few samples in a class for stratification
    # For regression, stratification is not directly applicable in train_test_split by y,
    # but ensure there are enough samples.
    stratify_opt = None
    # A simple check, if classification-like problem with few samples per class, stratify might be an issue
    # For regression, usually not an issue unless VALIDATION_SET_SIZE is too large for small datasets
    if len(np.unique(y_train_targets_full)) < len(y_train_targets_full) and len(np.unique(y_train_targets_full)) > 1: # Potentially classification
        # Check if any class count is 1, which would break stratify
        unique_ys, counts_ys = np.unique(y_train_targets_full, return_counts=True)
        if np.any(counts_ys == 1) and VALIDATION_SET_SIZE > 0 : # if any class has only one sample
           pass # stratify = None (default) is fine
        elif VALIDATION_SET_SIZE > 0:
           stratify_opt = y_train_targets_full


    X_hp_train, X_hp_val, y_hp_train, y_hp_val = train_test_split(
        X_list_train_scaled_full, y_train_targets_full,
        test_size=VALIDATION_SET_SIZE,
        random_state=GLOBAL_PARAMS.get("seed", 1234),
        shuffle=True,
        stratify=stratify_opt # Use stratify if applicable and safe
    )
    print(f"HP-Search Training set size: {len(X_hp_train)} sequences")
    print(f"HP-Search Validation set size: {len(X_hp_val)} sequences")

    if len(X_hp_train) == 0 or len(X_hp_val) == 0:
        print("Error: Training or validation set for HP search is empty after split. Adjust VALIDATION_SET_SIZE or check data. Exiting.")
        exit()


    print("\n--- Initializing Hyperparameter Search (MLP Prediction Readout) ---")
    hp_search_optimizer = PredictionHPSearch(
        input_dim=input_dim,
        X_hp_train_data=X_hp_train, y_hp_train_data=y_hp_train,
        X_hp_val_data=X_hp_val, y_hp_val_data=y_hp_val,
        n_iterations=N_HP_ITERATIONS * n_layers,
        n_reservoirs=n_layers,
        reservoir_hp_space=SEARCH_SPACE,
        global_params=GLOBAL_PARAMS,
        epsilon_greedy=0.3
    )

    start_hp_search = time.time()
    hp_search_optimizer.search()
    end_hp_search = time.time()
    results = {}

    print("\n--- Hyperparameter Search Finished ---")
    if not hp_search_optimizer.best_params_list or hp_search_optimizer.best_reservoir_model is None: # Added check for best_reservoir_model
        print("No best model/params found from HP search. Exiting."); exit()

    print(f"Best Score (neg validation MSE): {hp_search_optimizer.best_score:.5f}")
    if hp_search_optimizer.best_score != -np.inf:
        print(f"Corresponds to best avg validation MSE/sample: {-hp_search_optimizer.best_score:.5f}")
    for i, params in enumerate(hp_search_optimizer.best_params_list): print(f" Layer {i+1}: {params}")

    best_reservoirs_list, best_mlp_readout_from_hps, mlp_training_time_hp_best = hp_search_optimizer.best_reservoir_model
    results["mlp_training_time_hps_best_model"] = mlp_training_time_hp_best

    print("\n--- Retraining Best Model on Full Training Data (MLP Prediction, Aggregated) ---")
    warmup_final = GLOBAL_PARAMS.get("warmup", 0)

    print("Preparing full training data for final MLP (using all original training data)...")
    # Use the full original scaled training data for the final model
    final_mlp_train_inputs, final_mlp_train_targets = hp_search_optimizer._get_aggregated_reservoir_outputs_and_targets(
        X_list_train_scaled_full, y_train_targets_full, # Use the complete original training set
        best_reservoirs_list, warmup_final
    )

    if final_mlp_train_inputs is None or final_mlp_train_inputs.shape[0] < 1:
        print("Could not generate aggregated data from full training set for final MLP. Exiting.")
        exit()

    if final_mlp_train_targets.ndim > 1 and final_mlp_train_targets.shape[1] == 1:
        final_mlp_train_targets = final_mlp_train_targets.ravel()

    final_mlp = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                             activation=MLP_ACTIVATION, solver=MLP_SOLVER,
                             max_iter=MLP_MAX_ITER * 2, # More iterations for final model
                             early_stopping=MLP_EARLY_STOPPING,
                             n_iter_no_change=MLP_N_ITER_NO_CHANGE,
                             random_state=GLOBAL_PARAMS.get("seed"), alpha=MLP_ALPHA)
    print("Training final MLP readout on full training data...")
    start_final_mlp_train = time.time()
    try:
        final_mlp.fit(final_mlp_train_inputs, final_mlp_train_targets)
    except ValueError as e: # Fallback if too few samples for early stopping in the full train set (unlikely but possible)
        print(f"Final MLP fitting error: {e}. Trying without early stopping.")
        final_mlp_no_es = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                                       activation=MLP_ACTIVATION, solver=MLP_SOLVER,
                                       max_iter=MLP_MAX_ITER * 10,
                                       random_state=GLOBAL_PARAMS.get("seed"), alpha=MLP_ALPHA)
        final_mlp_no_es.fit(final_mlp_train_inputs, final_mlp_train_targets)
        final_mlp = final_mlp_no_es # reassign
    end_final_mlp_train = time.time()
    results["final_mlp_training_time"] = end_final_mlp_train - start_final_mlp_train
    results["final_mlp_training_start_time"] = start_final_mlp_train
    results["final_mlp_training_end_time"] = end_final_mlp_train
    print(f"Final MLP training time: {results['final_mlp_training_time']:.2f}s")

    mls_macs, n_params_mlp = count_sklearn_mlp_macs_and_params(final_mlp)
    results["mlp_macs"] = mls_macs
    results["n_model_parameters_mlp_readout"] = n_params_mlp

    results["final_mlp_prediction_time_test"] = None
    results["RC_aggregation_test_time"] = None
    results["r_squared"] = None
    results["rmse"] = None

    if X_list_test_scaled and y_test_targets is not None and len(y_test_targets) > 0:
        print("Preparing test data for final MLP evaluation...")
        start_RC_aggregation_test = time.time()
        final_mlp_test_inputs, final_mlp_test_actual_targets = hp_search_optimizer._get_aggregated_reservoir_outputs_and_targets(
            X_list_test_scaled, y_test_targets,
            best_reservoirs_list, warmup_final
        )
        end_RC_aggregation_test = time.time()
        results["RC_aggregation_test_time"] = end_RC_aggregation_test - start_RC_aggregation_test

        if final_mlp_test_inputs is None or final_mlp_test_inputs.shape[0] == 0:
            print("Could not generate aggregated data from test set for final MLP evaluation.")
        else:
            if final_mlp_test_actual_targets.ndim > 1 and final_mlp_test_actual_targets.shape[1] == 1:
                final_mlp_test_actual_targets = final_mlp_test_actual_targets.ravel()
            
            if len(final_mlp_test_actual_targets) == 0:
                print("Test targets are empty after processing. Skipping test evaluation.")
            else:
                print("Predicting with final MLP on test data...")
                start_final_mlp_predict_test = time.time()
                test_predictions = final_mlp.predict(final_mlp_test_inputs)
                end_final_mlp_predict_test = time.time()
                results["final_mlp_prediction_time_test"] = end_final_mlp_predict_test - start_final_mlp_predict_test
                results["final_mlp_prediction_start_time_test"] = start_final_mlp_predict_test
                results["final_mlp_prediction_end_time_test"] = end_final_mlp_predict_test
                print(f"Final MLP prediction time on test set: {results['final_mlp_prediction_time_test']:.4f}s")

                if len(test_predictions) != len(final_mlp_test_actual_targets):
                    print("Mismatch in length between test predictions and actual targets. Skipping scoring.")
                else:
                    r2_final = r2_score(final_mlp_test_actual_targets, test_predictions)
                    rmse_final = np.sqrt(mean_squared_error(final_mlp_test_actual_targets, test_predictions))

                    print("\n--- Final Model Performance on Test Set (Prediction) ---")
                    print(f" R-squared (R²): {r2_final:.4f}")
                    print(f" Root Mean Squared Error (RMSE): {rmse_final:.4f}")

                    results.update({
                        "r_squared": r2_final,
                        "rmse": rmse_final,
                    })
    else:
        print("Skipping final test evaluation as no valid/sufficient test data or targets were available.")

    results["hp_search_time"] = end_hp_search - start_hp_search
    results["model_name"] = f"RCPredNet_Aggregated_MLP_{ASSET_NAME}_ValSplit"
    n_params_mlp = sum(c.size for c in final_mlp.coefs_) + sum(i.size for i in final_mlp.intercepts_) if hasattr(final_mlp, 'coefs_') else 0
    results["n_model_parameters_readout"] = n_params_mlp
    results["n_model_parameters_total"] = n_params_mlp # Add reservoir params if desired
    results["input_dim"] = input_dim
    results["n_layers_rc"] = n_layers
    results["validation_set_size_hps"] = VALIDATION_SET_SIZE


    results_path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_layers}_layers", f"{ASSET_NAME}_prediction_results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    save_csv_results(results, results_path)

    print(f"\n--- RC Prediction (Aggregated MLP Readout with Validation Split) Task for {ASSET_NAME} Finished ---")

    path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_layers}_layers", f"{ASSET_NAME}_prediction_time.json")
    prediction_time_dict = {
        "start_time_hp_search": start_hp_search, "end_time_hp_search": end_hp_search,
        "total_time_hp_search": end_hp_search - start_hp_search,
        "start_time_mlp_training": start_final_mlp_train, "end_time_mlp_training": end_final_mlp_train,
        "total_mlp_training_time": results["final_mlp_training_time"],
        "final_mlp_prediction_time_test": results.get("final_mlp_prediction_time_test"), # Use .get for safety
        "RC_aggregation_test_time": results.get("RC_aggregation_test_time"),
    }
    with open(path, 'w') as f: json.dump(prediction_time_dict, f, indent=4)
    print(f"Prediction time saved to {path}")

    hps_path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_layers}_layers", f"{ASSET_NAME}_rc_hyperparameters.json")
    best_hps_serializable = {}
    if hp_search_optimizer.best_params_list: # Check if list is not empty
        for i, params_item in enumerate(hp_search_optimizer.best_params_list):
            serializable_params = {}
            if isinstance(params_item, dict):
                for key, value in params_item.items():
                    if isinstance(value, np.integer): serializable_params[key] = int(value)
                    elif isinstance(value, np.floating): serializable_params[key] = float(value)
                    elif isinstance(value, np.bool_): serializable_params[key] = bool(value)
                    elif isinstance(value, np.ndarray): serializable_params[key] = value.tolist()
                    else: serializable_params[key] = value
            else: # Fallback for non-dict params_item (should not happen with current structure)
                serializable_params = str(params_item) # Convert to string as a safe fallback
            best_hps_serializable[f"layer_{i+1}"] = serializable_params
    
    with open(hps_path, 'w') as f: json.dump(best_hps_serializable, f, indent=4)
    print(f"Best hyperparameters saved to {hps_path}")