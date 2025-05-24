import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from reservoirpy.nodes import Reservoir, IPReservoir
from reservoirpy.mat_gen import uniform, bernoulli
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import traceback
from reservoirpy.utils import verbosity
import sys
import time
import pandas as pd
import json
import pickle

verbosity(0)

# === PYTORCH DNN & GPU CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# === PREDICTION SCRIPT CONFIGS FOR FD01 DATA ===
# SR = 16000 # Removed
# N_MFCC = 13 # Removed, will be derived from data

# --- PyTorch DNN Configuration ---
DNN_HIDDEN_LAYER_SIZES = (64, 32)
DNN_ACTIVATION = nn.ReLU()
DNN_LEARNING_RATE = 0.001
DNN_MAX_EPOCHS = 500 # Increased epochs for PyTorch training
DNN_BATCH_SIZE = 32
DNN_EARLY_STOPPING = True
DNN_N_ITER_NO_CHANGE = 20 # Increased patience for PyTorch early stopping
DNN_ALPHA = 0.0001 # L2 regularization (weight_decay in Adam)

# --- Define PyTorch DNN Model ---
class TorchMLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size=1):
        super(TorchMLP, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(DNN_ACTIVATION)
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def count_torch_dnn_macs_and_params(model, input_shape):
    """
    Calculates the total MACs and parameters for a PyTorch nn.Module.
    Note: This is an approximation, especially for activations.
    Focuses on Linear layers.
    """
    total_macs = 0
    total_params = 0
    
    # Use a dummy input to trace the model (if needed for complex models, but here we can inspect layers)
    # We will manually iterate through linear layers for simplicity and clarity
    
    for layer in model.network:
        if isinstance(layer, nn.Linear):
            # MACs = input_features * output_features
            total_macs += layer.in_features * layer.out_features
            # Params = weights + biases
            total_params += layer.weight.nelement() + layer.bias.nelement()
            
    return total_macs, total_params

def get_Hps_folder(asset_name, n_layers):
    HPS_FOLDER = f"./RC_selected_hps/{asset_name}/mlp_aggregated_prediction_step1/"
    return HPS_FOLDER

def getResultsFolder(asset_name):
    RESULTS_FOLDER = f"./time_only_torch/NASA_results_val_selected_hps_{asset_name}/race_free_torch_dnn_aggregated_prediction_val_split_step1/" # Updated folder name
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
    hps_file_path = os.path.join(results_folder_func(asset_name, n_layers), f"{n_layers}_layers", f"{asset_name}_rc_hyperparameters.json")
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

def get_aggregated_reservoir_outputs_and_targets(X_data, Y_targets_data, reservoirs, warmup_glob, input_dim_for_reservoirs):
    aggregated_inputs = []
    aggregated_targets = []

    if not X_data or len(X_data) != len(Y_targets_data):
        print("Error: Mismatch between number of input sequences and target values, or X_data is empty.")
        return None, None

    for i, x_seq_single_orig in enumerate(X_data):
        y_target_for_sequence = Y_targets_data[i]

        if x_seq_single_orig.shape[1] != input_dim_for_reservoirs:
            print(f"Warning: Sequence {i} has {x_seq_single_orig.shape[1]} features, expected {input_dim_for_reservoirs}. Skipping.")
            continue

        for res_node in reservoirs:
            res_node.reset()

        if len(x_seq_single_orig) == 0: continue
        
        x_seq_single = x_seq_single_orig.copy()

        effective_inputs = x_seq_single
        if warmup_glob > 0 and len(x_seq_single) > warmup_glob :
            input_warmup = x_seq_single[:warmup_glob]
            current_layer_warmup = input_warmup
            for res_idx, res_node in enumerate(reservoirs):
                if current_layer_warmup.shape[1] != res_node.input_dim:
                     print(f"Warmup Dimension mismatch for res {res_idx}: expected {res_node.input_dim}, got {current_layer_warmup.shape[1]}")
                     break
                _ = res_node.run(current_layer_warmup, reset=False)
                current_layer_warmup = _
                if current_layer_warmup.shape[0] == 0: break
            effective_inputs = x_seq_single[warmup_glob:]
        elif warmup_glob > 0 and len(x_seq_single) <= warmup_glob:
            pass

        if effective_inputs.shape[0] == 0:
            current_sequence_data_fb = x_seq_single
            final_layer_output_sequence = None
            for res_idx, res_node in enumerate(reservoirs):
                if current_sequence_data_fb.shape[1] != res_node.input_dim:
                     print(f"Fallback Dimension mismatch for res {res_idx}: expected {res_node.input_dim}, got {current_sequence_data_fb.shape[1]}")
                     final_layer_output_sequence = None
                     break
                output_states_sequence_fb = res_node.run(current_sequence_data_fb, reset=False)
                current_sequence_data_fb = output_states_sequence_fb
                if current_sequence_data_fb.shape[0] == 0:
                    final_layer_output_sequence = None
                    break
                final_layer_output_sequence = current_sequence_data_fb
            if final_layer_output_sequence is None or final_layer_output_sequence.shape[0] == 0 :
                continue
            final_layer_output_for_aggregation = final_layer_output_sequence[-1:,:]

        else:
            current_sequence_data = effective_inputs
            final_layer_output_sequence = None
            for res_idx, res_node in enumerate(reservoirs):
                if current_sequence_data.shape[1] != res_node.input_dim:
                     print(f"Main Run Dimension mismatch for res {res_idx}: expected {res_node.input_dim}, got {current_sequence_data.shape[1]}")
                     final_layer_output_sequence = None
                     break
                output_states_sequence = res_node.run(current_sequence_data, reset=False)
                current_sequence_data = output_states_sequence
                if current_sequence_data.shape[0] == 0:
                    final_layer_output_sequence = None
                    break
                final_layer_output_sequence = current_sequence_data
            
            if final_layer_output_sequence is None or final_layer_output_sequence.shape[0] == 0 :
                continue
            final_layer_output_for_aggregation = final_layer_output_sequence

        aggregated_reservoir_output_for_sequence = np.mean(final_layer_output_for_aggregation, axis=0)
        aggregated_inputs.append(aggregated_reservoir_output_for_sequence)
        aggregated_targets.append(y_target_for_sequence)

    if not aggregated_inputs:
        return None, None

    return np.array(aggregated_inputs), np.array(aggregated_targets)

def train_torch_dnn(model, X_train, y_train, X_val=None, y_val=None):
    model.to(DEVICE)
    
    # Ensure y_train is 2D for loss calculation if needed, or 1D
    y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(dataset=train_dataset, batch_size=DNN_BATCH_SIZE, shuffle=True)
    
    use_validation = X_val is not None and y_val is not None
    if use_validation:
        y_val = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        val_loader = DataLoader(dataset=val_dataset, batch_size=DNN_BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=DNN_LEARNING_RATE, weight_decay=DNN_ALPHA)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    start_time = time.time()
    
    for epoch in range(DNN_MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)

        # Validation Step
        if use_validation:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)
            
            # print(f"Epoch {epoch+1}/{DNN_MAX_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early Stopping Check
            if DNN_EARLY_STOPPING:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Optionally save the best model state
                    # torch.save(model.state_dict(), 'best_model.pth')
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= DNN_N_ITER_NO_CHANGE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        # else:
            # print(f"Epoch {epoch+1}/{DNN_MAX_EPOCHS}, Train Loss: {train_loss:.4f}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"DNN training finished. Time: {training_time:.2f}s")
    return model, training_time, start_time, end_time

def predict_torch_dnn(model, X_test):
    model.to(DEVICE)
    model.eval()
    
    test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    
    predictions = []
    start_time = time.time()
    with torch.no_grad():
        # Process in batches if X_test is very large, for now, process all at once
        outputs = model(test_tensor)
        predictions = outputs.cpu().numpy().ravel() # Move back to CPU and flatten
    end_time = time.time()
    prediction_time = end_time - start_time
    print(f"DNN prediction time: {prediction_time:.4f}s")
    return predictions, prediction_time, start_time, end_time


# === MAIN SCRIPT LOGIC ===
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Please provide the number of layers and the asset name.")

    n_layers = int(sys.argv[1])
    ASSET_NAME = sys.argv[2]
    print(f"Number of layers: {n_layers}")
    print(f"Using data asset: {ASSET_NAME}")

    overall_start_time = time.time()

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

    y_train_targets_full = np.array(y_train_targets_full).ravel()

    print("\n--- Loading Hyperparameters from JSON ---")
    start_hp_load_time = time.time()
    loaded_hyperparameters = load_hyperparameters_from_json(ASSET_NAME, n_layers, get_Hps_folder)
    end_hp_load_time = time.time()
    hp_loading_duration = end_hp_load_time - start_hp_load_time
    print(f"Hyperparameters loaded in {hp_loading_duration:.4f}s")
    for i, params in enumerate(loaded_hyperparameters): print(f" Layer {i+1}: {params}")

    results = {}
    results["hp_loading_time"] = hp_loading_duration

    print("\n--- Building Reservoirs with Loaded Hyperparameters ---")
    reservoir_seed = GLOBAL_PARAMS.get("seed", None)
    warmup_final = GLOBAL_PARAMS.get("warmup", 0)

    reservoirs_list = []
    current_input_dim_for_layer = input_dim

    for i, layer_hp in enumerate(loaded_hyperparameters):
        reservoir_node_type = layer_hp.get("RC_node_type", "ESN")
        _activation = layer_hp.get("activation", "tanh")
        
        required_hps = ["units", "sr", "lr", "input_scaling"]
        for hp_name in required_hps:
            if hp_name not in layer_hp:
                print(f"ERROR: Missing required HP '{hp_name}' in loaded HPs for layer {i+1}. Content: {layer_hp}")
                sys.exit(1)

        shared_params = dict(
            units=layer_hp["units"], input_dim=current_input_dim_for_layer,
            sr=layer_hp["sr"], lr=layer_hp["lr"],
            input_scaling=layer_hp["input_scaling"],
            rc_connectivity=layer_hp.get("connectivity"),
            activation=_activation,
            W=uniform(high=1.0, low=-1.0),
            Win=bernoulli,
            seed=reservoir_seed + i if reservoir_seed is not None else None
        )
        if reservoir_node_type == "IPESN":
            node = IPReservoir(**shared_params,
                              mu=layer_hp.get("mu", 0.0),
                              learning_rate=layer_hp.get("learning_rate", 1e-3),
                              epochs=GLOBAL_PARAMS.get("epochs", 1))
        else:
            node = Reservoir(**shared_params)
        reservoirs_list.append(node)
        current_input_dim_for_layer = layer_hp["units"]

    print(f"Built {len(reservoirs_list)} reservoir layers.")
    
    reservoir_output_dim = current_input_dim_for_layer # This is the input dim for the DNN

    print("\n--- Training Final Model on Full Training Data (Torch DNN Prediction, Aggregated) ---")
    print("Preparing full training data for final DNN...")
    final_dnn_train_inputs, final_dnn_train_targets = get_aggregated_reservoir_outputs_and_targets(
        X_list_train_scaled_full, y_train_targets_full,
        reservoirs_list, warmup_final, input_dim
    )

    if final_dnn_train_inputs is None or final_dnn_train_inputs.shape[0] < 1:
        print("Could not generate aggregated data from full training set for final DNN. Exiting.")
        exit()

    # Instantiate the PyTorch model
    final_dnn = TorchMLP(input_size=reservoir_output_dim, hidden_layer_sizes=DNN_HIDDEN_LAYER_SIZES)
    print(final_dnn)

    print("Training final Torch DNN readout on full training data...")
    # Train the model (No validation split here, using full training data)
    final_dnn, train_time, start_train_time, end_train_time = train_torch_dnn(
        final_dnn, final_dnn_train_inputs, final_dnn_train_targets, None, None
    )

    # Save the final DNN model
    final_dnn_model_path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_layers}_layers", f"{ASSET_NAME}_final_torch_dnn_model.pth")
    os.makedirs(os.path.dirname(final_dnn_model_path), exist_ok=True)
    torch.save(final_dnn.state_dict(), final_dnn_model_path)
    results["final_dnn_model_size_bytes"] = os.path.getsize(final_dnn_model_path)

    results["final_dnn_training_time"] = train_time
    results["final_dnn_training_start_time"] = start_train_time
    results["final_dnn_training_end_time"] = end_train_time
    print(f"Final DNN training time: {results['final_dnn_training_time']:.2f}s")

    dnn_macs, n_params_dnn = count_torch_dnn_macs_and_params(final_dnn, (1, reservoir_output_dim))
    results["dnn_macs"] = dnn_macs
    results["n_model_parameters_dnn_readout"] = n_params_dnn

    results["final_dnn_prediction_time_test"] = None
    results["RC_aggregation_test_time"] = None
    results["r_squared"] = None
    results["rmse"] = None

    if X_list_test_scaled and y_test_targets is not None and len(y_test_targets) > 0:
        print("Preparing test data for final DNN evaluation...")
        start_RC_aggregation_test = time.time()
        final_dnn_test_inputs, final_dnn_test_actual_targets = get_aggregated_reservoir_outputs_and_targets(
            X_list_test_scaled, y_test_targets,
            reservoirs_list, warmup_final, input_dim
        )
        end_RC_aggregation_test = time.time()
        results["RC_aggregation_test_time"] = end_RC_aggregation_test - start_RC_aggregation_test

        if final_dnn_test_inputs is None or final_dnn_test_inputs.shape[0] == 0:
            print("Could not generate aggregated data from test set for final DNN evaluation.")
        else:
            final_dnn_test_actual_targets = final_dnn_test_actual_targets.ravel()
            
            if len(final_dnn_test_actual_targets) == 0:
                print("Test targets are empty after processing. Skipping test evaluation.")
            else:
                print("Predicting with final DNN on test data...")
                test_predictions, pred_time, start_pred_time, end_pred_time = predict_torch_dnn(final_dnn, final_dnn_test_inputs)
                results["final_dnn_prediction_time_test"] = pred_time
                results["final_dnn_prediction_start_time_test"] = start_pred_time
                results["final_dnn_prediction_end_time_test"] = end_pred_time
                print(f"Final DNN prediction time on test set: {results['final_dnn_prediction_time_test']:.4f}s")

                if len(test_predictions) != len(final_dnn_test_actual_targets):
                    print(f"Mismatch in length between test predictions ({len(test_predictions)}) and actual targets ({len(final_dnn_test_actual_targets)}). Skipping scoring.")
                else:
                    r2_final = r2_score(final_dnn_test_actual_targets, test_predictions)
                    rmse_final = np.sqrt(mean_squared_error(final_dnn_test_actual_targets, test_predictions))

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
    results["model_name"] = f"RCPredNet_Aggregated_TorchDNN_{ASSET_NAME}_LoadedHPs" # Updated model name
    results["n_model_parameters_readout"] = n_params_dnn
    
    n_params_rc = 0
    for res_node in reservoirs_list:
        if hasattr(res_node, 'W') and res_node.W is not None: n_params_rc += res_node.W.size
        if hasattr(res_node, 'Win') and res_node.Win is not None: n_params_rc += res_node.Win.size
        if hasattr(res_node, 'Wfb') and res_node.Wfb is not None: n_params_rc += res_node.Wfb.size
        if hasattr(res_node, 'bias') and res_node.bias is not None: n_params_rc += res_node.bias.size
    results["n_model_parameters_reservoirs"] = n_params_rc
    results["n_model_parameters_total"] = n_params_dnn + n_params_rc
    
    results["input_dim"] = input_dim
    results["n_layers_rc"] = n_layers
    results["validation_set_size_hps"] = "N/A (HPs Loaded)"
    results["device_used"] = str(DEVICE)


    results_path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_layers}_layers", f"{ASSET_NAME}_prediction_results_loadedHPs_torch.csv") # New CSV name
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results['loaded_hyperparameters'] = json.dumps(loaded_hyperparameters)
    save_csv_results(results, results_path)

    print(f"\n--- RC Prediction (Aggregated Torch DNN Readout with Loaded HPs) Task for {ASSET_NAME} Finished ---")

    path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_layers}_layers", f"{ASSET_NAME}_prediction_time_loadedHPs_torch.json") # New JSON name
    prediction_time_dict = {
        "hp_loading_time": hp_loading_duration,
        "start_time_dnn_training": start_train_time, "end_time_dnn_training": end_train_time,
        "total_dnn_training_time": results["final_dnn_training_time"],
        "final_dnn_prediction_time_test": results.get("final_dnn_prediction_time_test"),
        "RC_aggregation_test_time": results.get("RC_aggregation_test_time"),
        "total_script_execution_time": results["total_script_execution_time"],
        "overall_script_start_time_epoch": overall_start_time,
        "overall_script_end_time_epoch": overall_end_time,
        "device_used": str(DEVICE)
    }
    with open(path, 'w') as f: json.dump(prediction_time_dict, f, indent=4)
    print(f"Prediction time detail saved to {path}")