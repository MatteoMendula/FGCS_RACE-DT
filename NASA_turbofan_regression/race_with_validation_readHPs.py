import os
import numpy as np
# import librosa # Removed
from reservoirpy.nodes import Reservoir, IPReservoir
from reservoirpy.mat_gen import uniform, bernoulli
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import train_test_split # No longer needed for HP search validation split
import traceback
from reservoirpy.utils import verbosity
# from myRLS import RLS # RLS is no longer used
# BaseEpsilonGreedyMultiReservoirHPSearch is removed
import sys
import time
import pandas as pd
import json
import pickle

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
        return 0, 0 # Return 0 for params as well

    for coef_matrix in mlp_model.coefs_:
        macs_layer = coef_matrix.shape[0] * coef_matrix.shape[1]
        total_macs += macs_layer

    n_params_mlp = sum(c.size for c in mlp_model.coefs_) + sum(i.size for i in mlp_model.intercepts_) if hasattr(mlp_model, 'coefs_') else 0
    return total_macs, n_params_mlp

def get_Hps_folder(asset_name, n_layers):
    HPS_FOLDER = f"./RC_selected_hps/{asset_name}/mlp_aggregated_prediction_step2/" # Kept folder name assuming HPs were derived this way
    return HPS_FOLDER

def getResultsFolder(asset_name):
    RESULTS_FOLDER = f"./time_only/NASA_results_val_selected_hps_{asset_name}/race_free_mlp_aggregated_prediction_val_split_step2/" # Kept folder name assuming HPs were derived this way
    return RESULTS_FOLDER

# --- .npy File Paths ---
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
MLP_MAX_ITER = 300 # Max iter for final model training might be increased later
MLP_EARLY_STOPPING = True
MLP_N_ITER_NO_CHANGE = 10
MLP_ALPHA = 0.0001

# --- HYPERPARAMETER SEARCH CONFIGURATION (REMOVED) ---
# SEARCH_SPACE, N_HP_ITERATIONS, VALIDATION_SET_SIZE are removed.

GLOBAL_PARAMS = {
    "seed": 1234,
    "warmup": 0,
    "epochs": 100, # For IPESN
}

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

def load_hyperparameters_from_json(asset_name, n_layers, results_folder_func):
    hps_file_path = os.path.join(results_folder_func(asset_name, n_layers), f"{n_layers}_layers", f"best_hyperparameters_val_selected_{asset_name}.json")
    if not os.path.exists(hps_file_path):
        print(f"ERROR: Hyperparameter file not found at {hps_file_path}")
        print("Please ensure hyperparameters have been generated and saved by the original search script.")
        sys.exit(1)
    with open(hps_file_path, 'r') as f:
        loaded_hps_dict = json.load(f)
    
    loaded_params_list = []
    for i in range(1, n_layers + 1):
        layer_key = f"layer_{i}"
        if layer_key in loaded_hps_dict:
            loaded_params_list.append(loaded_hps_dict[layer_key])
        else:
            print(f"ERROR: Hyperparameter for {layer_key} not found in {hps_file_path}")
            sys.exit(1)
    print(f"Successfully loaded hyperparameters from {hps_file_path}")
    return loaded_params_list

# This function was a method in the removed PredictionHPSearch class.
# It's now a standalone utility function.
def get_aggregated_reservoir_outputs_and_targets(X_data, Y_targets_data, reservoirs, warmup_glob, input_dim_for_reservoirs):
    aggregated_inputs = []
    aggregated_targets = []

    if not X_data or len(X_data) != len(Y_targets_data):
        print("Error: Mismatch between number of input sequences and target values, or X_data is empty.")
        return None, None

    for i, x_seq_single_orig in enumerate(X_data):
        y_target_for_sequence = Y_targets_data[i]

        # Ensure the input sequence has the correct feature dimension for the first reservoir
        if x_seq_single_orig.shape[1] != input_dim_for_reservoirs:
            print(f"Warning: Sequence {i} has {x_seq_single_orig.shape[1]} features, expected {input_dim_for_reservoirs}. Skipping.")
            continue

        for res_node in reservoirs:
            res_node.reset()

        if len(x_seq_single_orig) == 0: continue
        
        x_seq_single = x_seq_single_orig.copy() # Work with a copy

        effective_inputs = x_seq_single
        if warmup_glob > 0 and len(x_seq_single) > warmup_glob :
            input_warmup = x_seq_single[:warmup_glob]
            current_layer_warmup = input_warmup
            for res_idx, res_node in enumerate(reservoirs):
                # print(f"Warmup: Res {res_idx}, input shape: {current_layer_warmup.shape}")
                if current_layer_warmup.shape[1] != res_node.input_dim:
                     print(f"Warmup Dimension mismatch for res {res_idx}: expected {res_node.input_dim}, got {current_layer_warmup.shape[1]}")
                     break # Cannot proceed with this reservoir chain for this sequence
                _ = res_node.run(current_layer_warmup, reset=False)
                current_layer_warmup = _
                if current_layer_warmup.shape[0] == 0: break
            effective_inputs = x_seq_single[warmup_glob:]
        elif warmup_glob > 0 and len(x_seq_single) <= warmup_glob:
            pass # Process the whole sequence if shorter than warmup

        if effective_inputs.shape[0] == 0:
            # This case might occur if warmup consumed the whole sequence,
            # or if the original sequence was shorter than warmup and we decided to use the full sequence.
            # We need a consistent way to get the last state.
            # Re-run the entire original sequence without slicing if it was meant to be processed as a whole.
            current_sequence_data_fb = x_seq_single # Use the full original sequence for fallback
            final_layer_output_sequence = None
            for res_idx, res_node in enumerate(reservoirs):
                # print(f"Fallback Run: Res {res_idx}, input shape: {current_sequence_data_fb.shape}")
                if current_sequence_data_fb.shape[1] != res_node.input_dim:
                     print(f"Fallback Dimension mismatch for res {res_idx}: expected {res_node.input_dim}, got {current_sequence_data_fb.shape[1]}")
                     final_layer_output_sequence = None # invalidate
                     break
                output_states_sequence_fb = res_node.run(current_sequence_data_fb, reset=False)
                current_sequence_data_fb = output_states_sequence_fb
                if current_sequence_data_fb.shape[0] == 0:
                    final_layer_output_sequence = None # invalidate
                    break
                final_layer_output_sequence = current_sequence_data_fb # Keep last valid output
            if final_layer_output_sequence is None or final_layer_output_sequence.shape[0] == 0 :
                continue
            final_layer_output_for_aggregation = final_layer_output_sequence[-1:,:]

        else: # Normal processing with effective_inputs
            current_sequence_data = effective_inputs
            final_layer_output_sequence = None
            for res_idx, res_node in enumerate(reservoirs):
                # print(f"Main Run: Res {res_idx}, input shape: {current_sequence_data.shape}")
                if current_sequence_data.shape[1] != res_node.input_dim:
                     print(f"Main Run Dimension mismatch for res {res_idx}: expected {res_node.input_dim}, got {current_sequence_data.shape[1]}")
                     final_layer_output_sequence = None # invalidate
                     break
                output_states_sequence = res_node.run(current_sequence_data, reset=False)
                current_sequence_data = output_states_sequence
                if current_sequence_data.shape[0] == 0:
                    final_layer_output_sequence = None # invalidate
                    break
                final_layer_output_sequence = current_sequence_data # Keep last valid output
            
            if final_layer_output_sequence is None or final_layer_output_sequence.shape[0] == 0 :
                continue
            final_layer_output_for_aggregation = final_layer_output_sequence


        # Take the mean of all time steps of the output from the LAST reservoir for this sequence
        aggregated_reservoir_output_for_sequence = np.mean(final_layer_output_for_aggregation, axis=0)
        aggregated_inputs.append(aggregated_reservoir_output_for_sequence)
        aggregated_targets.append(y_target_for_sequence)

    if not aggregated_inputs:
        return None, None

    return np.array(aggregated_inputs), np.array(aggregated_targets)


# === MAIN SCRIPT LOGIC ===
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Please provide the number of layers and the asset name.")

    n_layers = int(sys.argv[1])
    ASSET_NAME = sys.argv[2]
    print(f"Number of layers: {n_layers}")
    print(f"Using data asset: {ASSET_NAME}")

    overall_start_time = time.time() # For overall script timing

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
    all_train_samples_for_scaler = np.vstack(X_list_train_raw)
    scaler_x = StandardScaler().fit(all_train_samples_for_scaler)
    X_list_train_scaled_full = [scaler_x.transform(x) for x in X_list_train_raw]

    X_list_test_scaled = []
    if X_list_test_raw:
        X_list_test_scaled = [scaler_x.transform(x) for x in X_list_test_raw]
        print(f"Test data scaled. Num sequences: {len(X_list_test_scaled)}")
    else:
        print("No test data loaded.")

    # Ensure y_train_targets_full is a 1D array
    y_train_targets_full = np.array(y_train_targets_full).ravel()

    # HP Search Validation Split is removed. We use full training data.

    print("\n--- Loading Hyperparameters from JSON ---")
    start_hp_load_time = time.time()
    # Pass the function getResultsFolder to allow load_hyperparameters_from_json to construct the path
    loaded_hyperparameters = load_hyperparameters_from_json(ASSET_NAME, n_layers, get_Hps_folder)
    end_hp_load_time = time.time()
    hp_loading_duration = end_hp_load_time - start_hp_load_time
    print(f"Hyperparameters loaded in {hp_loading_duration:.4f}s")
    for i, params in enumerate(loaded_hyperparameters): print(f" Layer {i+1}: {params}")

    results = {} # Initialize results dictionary
    results["hp_loading_time"] = hp_loading_duration

    print("\n--- Building Reservoirs with Loaded Hyperparameters ---")
    reservoir_seed = GLOBAL_PARAMS.get("seed", None)
    warmup_final = GLOBAL_PARAMS.get("warmup", 0)

    reservoirs_list = []
    current_input_dim_for_layer = input_dim

    for i, layer_hp in enumerate(loaded_hyperparameters):
        reservoir_node_type = layer_hp.get("RC_node_type", "ESN") # Default to ESN if not specified
        _activation = layer_hp.get("activation", "tanh") # Default activation
        
        # Ensure core HPs are present
        required_hps = ["units", "sr", "lr", "input_scaling"]
        for hp_name in required_hps:
            if hp_name not in layer_hp:
                print(f"ERROR: Missing required HP '{hp_name}' in loaded HPs for layer {i+1}. Content: {layer_hp}")
                sys.exit(1)

        shared_params = dict(
            units=layer_hp["units"], input_dim=current_input_dim_for_layer,
            sr=layer_hp["sr"], lr=layer_hp["lr"],
            input_scaling=layer_hp["input_scaling"],
            rc_connectivity=layer_hp.get("connectivity"), # Can be None, ReservoirPy handles it
            activation=_activation,
            W=uniform(high=1.0, low=-1.0), # Re-initialize weight matrices
            Win=bernoulli,                 # Re-initialize weight matrices
            seed=reservoir_seed + i if reservoir_seed is not None else None
        )
        if reservoir_node_type == "IPESN":
            node = IPReservoir(**shared_params,
                              mu=layer_hp.get("mu", 0.0), # Default mu if not in HPs
                              learning_rate=layer_hp.get("learning_rate", 1e-3), # Default IP learning rate
                              epochs=GLOBAL_PARAMS.get("epochs", 1)) # epochs from global_params
        else: # ESN
            node = Reservoir(**shared_params)
        reservoirs_list.append(node)
        current_input_dim_for_layer = layer_hp["units"] # Output of current becomes input for next

    print(f"Built {len(reservoirs_list)} reservoir layers.")

    print("\n--- Training Final Model on Full Training Data (MLP Prediction, Aggregated) ---")
    print("Preparing full training data for final MLP...")
    # Use the full original scaled training data
    final_mlp_train_inputs, final_mlp_train_targets = get_aggregated_reservoir_outputs_and_targets(
        X_list_train_scaled_full, y_train_targets_full,
        reservoirs_list, warmup_final, input_dim # Pass original input_dim for the first reservoir
    )

    if final_mlp_train_inputs is None or final_mlp_train_inputs.shape[0] < 1:
        print("Could not generate aggregated data from full training set for final MLP. Exiting.")
        exit()
    
    if final_mlp_train_inputs.shape[0] < 2 and MLP_EARLY_STOPPING:
        print("Warning: Very few samples for MLP training, disabling early stopping.")
        current_mlp_early_stopping = False
        current_mlp_n_iter_no_change = MLP_N_ITER_NO_CHANGE # Keep this, but it won't be used
    else:
        current_mlp_early_stopping = MLP_EARLY_STOPPING
        current_mlp_n_iter_no_change = MLP_N_ITER_NO_CHANGE


    if final_mlp_train_targets.ndim > 1 and final_mlp_train_targets.shape[1] == 1:
        final_mlp_train_targets = final_mlp_train_targets.ravel()

    final_mlp = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                             activation=MLP_ACTIVATION, solver=MLP_SOLVER,
                             max_iter=MLP_MAX_ITER * 2, # More iterations for final model
                             early_stopping=current_mlp_early_stopping,
                             n_iter_no_change=current_mlp_n_iter_no_change,
                             random_state=GLOBAL_PARAMS.get("seed"), alpha=MLP_ALPHA)
    print("Training final MLP readout on full training data...")
    start_final_mlp_train = time.time()
    try:
        final_mlp.fit(final_mlp_train_inputs, final_mlp_train_targets)
    except ValueError as e:
        print(f"Final MLP fitting error: {e}. Trying without early stopping if it was enabled.")
        # Fallback if too few samples for early stopping (even if check above missed it)
        # or other ValueError from fit
        final_mlp_no_es = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                                       activation=MLP_ACTIVATION, solver=MLP_SOLVER,
                                       max_iter=MLP_MAX_ITER * 10, # Even more iterations
                                       random_state=GLOBAL_PARAMS.get("seed"), alpha=MLP_ALPHA,
                                       early_stopping=False) # Explicitly disable
        final_mlp_no_es.fit(final_mlp_train_inputs, final_mlp_train_targets)
        final_mlp = final_mlp_no_es
    end_final_mlp_train = time.time()

    # save the final MLP model
    final_mlp_model_path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_layers}_layers", f"{ASSET_NAME}_final_mlp_model.pkl")
    os.makedirs(os.path.dirname(final_mlp_model_path), exist_ok=True)
    with open(final_mlp_model_path, 'wb') as f:
        pickle.dump(final_mlp, f)
    # get the model size in bytes
    results["final_mlp_model_size_bytes"] = os.path.getsize(final_mlp_model_path)

    results["final_mlp_training_time"] = end_final_mlp_train - start_final_mlp_train
    results["final_mlp_training_start_time"] = start_final_mlp_train
    results["final_mlp_training_end_time"] = end_final_mlp_train
    print(f"Final MLP training time: {results['final_mlp_training_time']:.2f}s")

    mlp_macs, n_params_mlp = count_sklearn_mlp_macs_and_params(final_mlp)
    results["mlp_macs"] = mlp_macs
    results["n_model_parameters_mlp_readout"] = n_params_mlp

    results["final_mlp_prediction_time_test"] = None
    results["RC_aggregation_test_time"] = None
    results["r_squared"] = None
    results["rmse"] = None

    if X_list_test_scaled and y_test_targets is not None and len(y_test_targets) > 0:
        print("Preparing test data for final MLP evaluation...")
        start_RC_aggregation_test = time.time()
        final_mlp_test_inputs, final_mlp_test_actual_targets = get_aggregated_reservoir_outputs_and_targets(
            X_list_test_scaled, y_test_targets,
            reservoirs_list, warmup_final, input_dim # Pass original input_dim
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

    overall_end_time = time.time()
    results["total_script_execution_time"] = overall_end_time - overall_start_time
    results["model_name"] = f"RCPredNet_Aggregated_MLP_{ASSET_NAME}_LoadedHPs" # Updated model name
    # n_params_mlp already calculated and stored
    results["n_model_parameters_readout"] = n_params_mlp # Same as n_model_parameters_mlp_readout
    
    # Calculate reservoir parameters (approximate, not including all internal states if complex)
    n_params_rc = 0
    for res_node in reservoirs_list:
        if hasattr(res_node, 'W') and res_node.W is not None: n_params_rc += res_node.W.size
        if hasattr(res_node, 'Win') and res_node.Win is not None: n_params_rc += res_node.Win.size
        if hasattr(res_node, 'Wfb') and res_node.Wfb is not None: n_params_rc += res_node.Wfb.size # If feedback enabled
        if hasattr(res_node, 'bias') and res_node.bias is not None: n_params_rc += res_node.bias.size # If bias enabled
    results["n_model_parameters_reservoirs"] = n_params_rc
    results["n_model_parameters_total"] = n_params_mlp + n_params_rc
    
    results["input_dim"] = input_dim
    results["n_layers_rc"] = n_layers
    results["validation_set_size_hps"] = "N/A (HPs Loaded)"


    results_path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_layers}_layers", f"{ASSET_NAME}_prediction_results_loadedHPs.csv") # New CSV name
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    # Convert all params in loaded_hyperparameters to string for robust CSV saving if they contain complex objects
    results['loaded_hyperparameters'] = json.dumps(loaded_hyperparameters)
    save_csv_results(results, results_path)

    print(f"\n--- RC Prediction (Aggregated MLP Readout with Loaded HPs) Task for {ASSET_NAME} Finished ---")

    # Prediction time JSON saving
    path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_layers}_layers", f"{ASSET_NAME}_prediction_time_loadedHPs.json") # New JSON name
    prediction_time_dict = {
        "hp_loading_time": hp_loading_duration,
        "start_time_mlp_training": start_final_mlp_train, "end_time_mlp_training": end_final_mlp_train,
        "total_mlp_training_time": results["final_mlp_training_time"],
        "final_mlp_prediction_time_test": results.get("final_mlp_prediction_time_test"),
        "RC_aggregation_test_time": results.get("RC_aggregation_test_time"),
        "total_script_execution_time": results["total_script_execution_time"],
        "overall_script_start_time_epoch": overall_start_time,
        "overall_script_end_time_epoch": overall_end_time
    }
    with open(path, 'w') as f: json.dump(prediction_time_dict, f, indent=4)
    print(f"Prediction time detail saved to {path}")

    # Hyperparameters are loaded, not searched, so saving them again is optional.
    # The loaded HPs are already in the results CSV.
    # If you want to save the loaded HPs in their original JSON structure again for some reason:
    # loaded_hps_output_path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_layers}_layers", f"{ASSET_NAME}_loaded_rc_hyperparameters_echo.json")
    # with open(loaded_hps_output_path, 'w') as f:
    # json.dump(loaded_hyperparameters_dict_form, f, indent=4) # Assuming you have loaded_hps_dict from load function
    # print(f"Loaded hyperparameters (echo) saved to {loaded_hps_output_path}")