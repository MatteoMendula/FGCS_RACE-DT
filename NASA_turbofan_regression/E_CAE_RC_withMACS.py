import os
import numpy as np
from reservoirpy.nodes import Reservoir, IPReservoir
from reservoirpy.mat_gen import uniform, bernoulli
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
import traceback
from reservoirpy.utils import verbosity
from baseEpsilonGreedyMultiReservoirHPSearch import BaseEpsilonGreedyMultiReservoirHPSearch # Assuming this base class is available
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split # Import for splitting data

# Import PyTorch for CAE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

verbosity(0)

# Determine device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {DEVICE}")


# === CAE CONFIGURATION (PyTorch) ===
CAE_LATENT_DIM = 32
CAE_CONV_FILTERS = [64, 32] # e.g. [c1, c2] means Encoder: In->c1, c1->c2, c2->Latent
CAE_KERNEL_SIZE = 3 # Must be odd for simple 'same' padding
CAE_ACTIVATION_FN = nn.ReLU() # PyTorch activation function
CAE_EPOCHS = 30 # Reduced for faster testing, adjust as needed
CAE_BATCH_SIZE = 32
CAE_PADDING_VALUE = 0.0

# === PREDICTION SCRIPT CONFIGS ===
VALIDATION_SPLIT_SIZE = 0.4 # Proportion of training data to use for validation during HPS

def getResultsFolder(asset_name):
    RESULTS_FOLDER = f"./NASA_results_val_selected_hps_{asset_name}/cae_pytorch_rc_mlp_aggregated_prediction_val_split_withMAC/"
    return RESULTS_FOLDER

def get_data_file_path(asset_name):
    DATA_FILE_PATHS = {
        "x_train": f'./old_data/x_train_{asset_name}.npy',
        "y_train": f'./old_data/y_train_{asset_name}.npy',
        "x_test": f'./old_data/x_test_{asset_name}.npy',
        "y_test": f'./old_data/y_test_{asset_name}.npy'
    }
    return DATA_FILE_PATHS

MLP_HIDDEN_LAYER_SIZES = (64, 32)
MLP_ACTIVATION = 'relu'
MLP_SOLVER = 'adam'
MLP_MAX_ITER = 300
MLP_EARLY_STOPPING = True
MLP_N_ITER_NO_CHANGE = 10
MLP_ALPHA = 0.0001

SEARCH_SPACE = {
    "units": [5, 50, 500],
    "sr": {"min": np.log(0.1), "max": np.log(1.5)},
    "lr": [0.1, 1.0],
    "input_scaling": {"min": np.log(0.05), "max": np.log(2.0)},
    "connectivity": [0.01, 0.5],
    "activation": ["tanh", "sigmoid"],
    "RC_node_type": ["ESN", "IPESN"],
    "mu": [0.0, 0.2],
    "learning_rate": {"min": np.log(1e-5), "max": np.log(1e-2)},
}

GLOBAL_PARAMS = {
    "seed": 1234,
    "warmup": 5,
    "epochs": 100, # Note: This is 'epochs' for IPESN, not CAE or MLP
}
N_HP_ITERATIONS = 30 # Reduced for faster testing

# Set seed for reproducibility
np.random.seed(GLOBAL_PARAMS["seed"])
torch.manual_seed(GLOBAL_PARAMS["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_PARAMS["seed"])


def save_csv_results(results, file_full_path):
    os.makedirs(os.path.dirname(file_full_path), exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(file_full_path, index=False)
    print(f"Results saved to {file_full_path}")

# --- PyTorch CAE Definition ---
class EncoderNet(nn.Module):
    def __init__(self, input_features, latent_dim, conv_filters, kernel_size, activation_fn):
        super(EncoderNet, self).__init__()
        layers = []
        current_features = input_features
        for filters in conv_filters:
            layers.append(nn.Conv1d(current_features, filters, kernel_size,
                                    padding=(kernel_size - 1) // 2)) # 'same' padding
            layers.append(activation_fn)
            current_features = filters
        # Last conv layer to latent space
        layers.append(nn.Conv1d(current_features, latent_dim, kernel_size,
                                padding=(kernel_size - 1) // 2))
        layers.append(activation_fn) # Activation for encoder output
        self.encoder_layers = nn.Sequential(*layers) # Renamed to avoid conflict

    def forward(self, x):
        return self.encoder_layers(x)

class DecoderNet(nn.Module):
    def __init__(self, output_features, latent_dim, conv_filters, kernel_size, activation_fn):
        super(DecoderNet, self).__init__()
        layers = []
        current_features = latent_dim
        # Reversed conv_filters for decoder (e.g., from [c1, c2] to [c2, c1])
        for filters in reversed(conv_filters):
            layers.append(nn.Conv1d(current_features, filters, kernel_size,
                                    padding=(kernel_size - 1) // 2))
            layers.append(activation_fn)
            current_features = filters
        # Last conv layer to output original features
        layers.append(nn.Conv1d(current_features, output_features, kernel_size,
                                padding=(kernel_size - 1) // 2))
        # Optional: final activation for decoder output, e.g., nn.Sigmoid() if data is normalized to [0,1]
        # Here, we assume raw output matching input scale.
        self.decoder_layers = nn.Sequential(*layers) # Renamed to avoid conflict

    def forward(self, x):
        return self.decoder_layers(x)

class CAE1D_PyTorch(nn.Module):
    def __init__(self, input_features, latent_dim, conv_filters, kernel_size, activation_fn):
        super(CAE1D_PyTorch, self).__init__()
        self.encoder = EncoderNet(input_features, latent_dim, conv_filters, kernel_size, activation_fn)
        self.decoder = DecoderNet(input_features, latent_dim, conv_filters, kernel_size, activation_fn)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

def pad_list_of_numpy_arrays_to_tensor(list_of_np_arrays, max_len, num_features, padding_value=0.0, device=DEVICE):
    padded_sequences = np.full((len(list_of_np_arrays), max_len, num_features), padding_value, dtype=np.float32)
    for i, seq in enumerate(list_of_np_arrays):
        seq_len = min(seq.shape[0], max_len)
        padded_sequences[i, :seq_len, :] = seq[:seq_len, :]
    return torch.from_numpy(padded_sequences).to(device)


def load_FD_data_as_sequences(file_paths):
    try:
        X_train_np = np.load(file_paths['x_train'])
        y_train_np = np.load(file_paths['y_train'])
        X_test_np = np.load(file_paths['x_test'])
        y_test_np = np.load(file_paths['y_test'])
    except FileNotFoundError as e:
        print(f"Error loading .npy files: {e}. Ensure all files are present: {file_paths}")
        return None, None, None, None, 0, 0

    print(f"Loaded X_train (full) shape: {X_train_np.shape}, y_train (full) shape: {y_train_np.shape}")
    print(f"Loaded X_test shape: {X_test_np.shape}, y_test shape: {y_test_np.shape}")

    original_features = 0
    max_timesteps_train = 0

    def process_X_data(X_np):
        X_list_raw = []
        current_original_features = 0
        current_max_timesteps = 0
        if X_np.ndim == 2: 
            X_list_raw = [X_np[i:i+1, :] for i in range(X_np.shape[0])] 
            if X_np.shape[0] > 0: current_original_features = X_np.shape[1]
            current_max_timesteps = 1
        elif X_np.ndim == 3: 
            X_list_raw = [X_np[i, :, :] for i in range(X_np.shape[0])]
            if X_np.shape[0] > 0: current_original_features = X_np.shape[2]
            if X_list_raw:
                valid_seqs = [s for s in X_list_raw if s.ndim == 2 and s.shape[0] > 0]
                current_max_timesteps = max(s.shape[0] for s in valid_seqs) if valid_seqs else 0
            else:
                current_max_timesteps = 0
        elif X_np.size == 0: 
             pass 
        else:
            raise ValueError(f"X_np has unsupported dimension {X_np.ndim}. Expected 2D or 3D.")
        return X_list_raw, current_original_features, current_max_timesteps

    X_list_train_raw, original_features, max_timesteps_train = process_X_data(X_train_np)
    X_list_test_raw, _, _ = process_X_data(X_test_np) 

    y_train_targets_raw = y_train_np.ravel()
    y_test_targets_raw = y_test_np.ravel() if y_test_np.size > 0 else np.array([])

    if len(X_list_train_raw) != len(y_train_targets_raw):
        raise ValueError("Mismatch in number of training samples and training targets.")
    if X_list_test_raw and y_test_targets_raw.size > 0 and len(X_list_test_raw) != len(y_test_targets_raw):
         if len(X_list_test_raw) > 0: # Only raise if X_list_test_raw is not empty
            raise ValueError("Mismatch in number of test samples and test targets.")

    return X_list_train_raw, y_train_targets_raw, X_list_test_raw, y_test_targets_raw, original_features, max_timesteps_train


# --- MACs Calculation Functions ---
def count_cae_macs(cae_model, input_sequence_length, initial_input_features):
    """
    Calculates the total MACs for the 1D Convolutional Autoencoder.
    Assumes 'same' padding for Conv1D layers, so sequence length remains constant through convolutions.
    """
    total_macs = 0
    
    # --- Encoder MACs ---
    current_in_channels = initial_input_features
    # Iterate through layers of the Sequential model within EncoderNet
    for layer in cae_model.encoder.encoder_layers:
        if isinstance(layer, nn.Conv1d):
            # MACs for Conv1D = output_channels * output_length * (kernel_size * input_channels)
            # output_length is input_sequence_length due to 'same' padding
            macs_layer = layer.out_channels * input_sequence_length * (layer.kernel_size[0] * layer.in_channels)
            total_macs += macs_layer
            current_in_channels = layer.out_channels # This becomes input_channels for next Conv1D
            
    # --- Decoder MACs ---
    # Input to decoder is latent_dim channels
    # The first Conv1d in decoder takes latent_dim as in_channels
    # The structure of DecoderNet needs to be known or inferred to trace channels
    
    # Let's trace current_in_channels starting from latent_dim for decoder
    current_in_channels = CAE_LATENT_DIM # From the encoder output
    for layer in cae_model.decoder.decoder_layers:
        if isinstance(layer, nn.Conv1d):
            macs_layer = layer.out_channels * input_sequence_length * (layer.kernel_size[0] * layer.in_channels)
            total_macs += macs_layer
            current_in_channels = layer.out_channels
            
    return total_macs

def count_sklearn_mlp_macs(mlp_model):
    """
    Calculates the total MACs for a scikit-learn MLPRegressor or MLPClassifier.
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
    return total_macs


class PredictionHPSearch(BaseEpsilonGreedyMultiReservoirHPSearch):
    def __init__(self, input_dim, X_train_list_encoded, y_train_targets,
                 X_val_list_encoded, y_val_targets, 
                 n_iterations, n_reservoirs, reservoir_hp_space, global_params,
                 epsilon_greedy=0.3):
        
        super().__init__(input_dim, X_train_list_encoded, y_train_targets,
                         X_val_list_encoded, y_val_targets, 
                         n_iterations, n_reservoirs, reservoir_hp_space, global_params, epsilon_greedy)
        
        self.X_val = X_val_list_encoded
        self.y_val = y_val_targets


    def _get_aggregated_reservoir_outputs_and_targets(self, X_data_encoded, Y_targets_data, reservoirs, warmup_glob):
        aggregated_inputs = []
        aggregated_targets = [] 

        if not X_data_encoded: 
            return np.array([]), np.array([])
        if len(X_data_encoded) != len(Y_targets_data):
            print(f"Error: Mismatch between number of input sequences ({len(X_data_encoded)}) and target values ({len(Y_targets_data)}).")
            return None, None

        for i, x_seq_encoded_single_np in enumerate(X_data_encoded): 
            y_target_for_sequence = Y_targets_data[i] 

            for res_node in reservoirs: 
                res_node.reset()
            
            if x_seq_encoded_single_np.shape[0] == 0: continue 
            
            effective_inputs = x_seq_encoded_single_np
            if warmup_glob > 0 and len(x_seq_encoded_single_np) > warmup_glob :
                input_warmup = x_seq_encoded_single_np[:warmup_glob]
                current_layer_warmup = input_warmup
                for res_node in reservoirs:
                    _ = res_node.run(current_layer_warmup, reset=False) 
                    current_layer_warmup = _
                    if current_layer_warmup.shape[0] == 0: break 
                effective_inputs = x_seq_encoded_single_np[warmup_glob:]
            elif warmup_glob > 0 and len(x_seq_encoded_single_np) <= warmup_glob:
                pass 

            final_layer_output_sequence = None 
            if effective_inputs.shape[0] == 0: 
                if len(x_seq_encoded_single_np) > 0: 
                    current_sequence_data_fb = x_seq_encoded_single_np
                    for res_node in reservoirs:
                        output_states_sequence_fb = res_node.run(current_sequence_data_fb, reset=False)
                        current_sequence_data_fb = output_states_sequence_fb
                        if current_sequence_data_fb.shape[0] == 0: break
                    if current_sequence_data_fb.shape[0] > 0:
                        final_layer_output_sequence = current_sequence_data_fb
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
            
            if final_layer_output_sequence is None or final_layer_output_sequence.shape[0] == 0: continue

            aggregated_reservoir_output_for_sequence = np.mean(final_layer_output_sequence, axis=0)
            
            aggregated_inputs.append(aggregated_reservoir_output_for_sequence)
            aggregated_targets.append(y_target_for_sequence)
            
        if not aggregated_inputs: 
            return np.array([]), np.array([])
            
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
            
            mlp_train_inputs_np, mlp_train_targets_np = self._get_aggregated_reservoir_outputs_and_targets(
                self.X_train, self.y_train, reservoirs, warmup_glob 
            )

            if mlp_train_inputs_np is None or mlp_train_inputs_np.shape[0] < 2:
                return -np.inf, None, None, None, None, None, None 
            
            if mlp_train_targets_np.ndim > 1 and mlp_train_targets_np.shape[1] == 1:
                mlp_train_targets_np = mlp_train_targets_np.ravel()

            mlp_readout = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                                       activation=MLP_ACTIVATION,
                                       solver=MLP_SOLVER,
                                       max_iter=MLP_MAX_ITER,
                                       early_stopping=MLP_EARLY_STOPPING,
                                       n_iter_no_change=MLP_N_ITER_NO_CHANGE,
                                       random_state=reservoir_seed,
                                       alpha=MLP_ALPHA)
            
            start_mlp_training_time = time.time()
            try:
                mlp_readout.fit(mlp_train_inputs_np, mlp_train_targets_np)
            except ValueError as e: 
                mlp_readout_no_es = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                                           activation=MLP_ACTIVATION, solver=MLP_SOLVER,
                                           max_iter=MLP_MAX_ITER * 10, random_state=reservoir_seed, alpha=MLP_ALPHA)
                mlp_readout_no_es.fit(mlp_train_inputs_np, mlp_train_targets_np)
                mlp_readout = mlp_readout_no_es
            end_mlp_training_time = time.time()
            mlp_training_duration = end_mlp_training_time - start_mlp_training_time

            mlp_val_inputs_np, mlp_val_targets_np = self._get_aggregated_reservoir_outputs_and_targets(
                self.X_val, self.y_val, reservoirs, warmup_glob 
            )

            if mlp_val_inputs_np is None or mlp_val_inputs_np.shape[0] == 0:
                return -np.inf, None, None, None, None, None, None 

            if mlp_val_targets_np.ndim > 1 and mlp_val_targets_np.shape[1] == 1:
                mlp_val_targets_np = mlp_val_targets_np.ravel()
            
            predicted_val_outputs = mlp_readout.predict(mlp_val_inputs_np)
            
            if len(mlp_val_targets_np) == 0 or len(predicted_val_outputs) == 0:
                return -np.inf, None, None, None, None, None, None

            validation_mse = mean_squared_error(mlp_val_targets_np, predicted_val_outputs)
            score = -validation_mse 
            
            model_tuple = (reservoirs, mlp_readout, mlp_training_duration) 
            output_dim_of_last_reservoir = reservoirs[-1].output_dim if reservoirs else None
            
            return float(score), None, model_tuple, output_dim_of_last_reservoir, None, None, None

        except Exception as e:
            print(f"ERROR during MLP (prediction) HPS evaluation with HPs {current_params_list}: {e}")
            traceback.print_exc()
            return -np.inf, None, None, None, None, None, None

def scale_data_list(data_list_raw, scaler, original_features, asset_name, data_name=""):
    if not data_list_raw:
        print(f"No {data_name} data for {asset_name} to scale.")
        return [], [] # ensure two values are always returned
    
    valid_seqs = [seq for seq in data_list_raw if seq.ndim == 2 and seq.shape[0] > 0 and seq.shape[1] == original_features]
    if not valid_seqs:
        print(f"No valid sequences in {data_name} data for {asset_name} for scaling.")
        return [], []
    
    scaled_list = [scaler.transform(x) for x in valid_seqs]
    print(f"{data_name} data for {asset_name} scaled. Num sequences: {len(scaled_list)}")
    return scaled_list, valid_seqs 

def encode_data_list(data_list_scaled, cae_encoder, max_timesteps_padding, original_features, asset_name, data_name=""):
    if not data_list_scaled:
        print(f"No scaled {data_name} data for {asset_name} to encode.")
        return []
    
    encoded_list = []
    with torch.no_grad():
        for x_seq_scaled_np in data_list_scaled:
            original_len = x_seq_scaled_np.shape[0]
            x_seq_padded_torch = pad_list_of_numpy_arrays_to_tensor(
                [x_seq_scaled_np], max_timesteps_padding, original_features, CAE_PADDING_VALUE, DEVICE
            )
            x_seq_cae_input = x_seq_padded_torch.permute(0, 2, 1)
            encoded_output_torch = cae_encoder(x_seq_cae_input)
            encoded_output_permuted = encoded_output_torch.permute(0, 2, 1)
            encoded_seq_np = encoded_output_permuted[0, :original_len, :].cpu().numpy()
            encoded_list.append(encoded_seq_np)
    print(f"{data_name} data for {asset_name} encoded. Num sequences: {len(encoded_list)}. Latent dim: {CAE_LATENT_DIM}")
    return encoded_list


# === MAIN SCRIPT LOGIC ===
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <n_rc_layers> <asset_name>")
        # Create a dummy BaseEpsilonGreedyMultiReservoirHPSearch if it's not found, for placeholder purposes
        if "BaseEpsilonGreedyMultiReservoirHPSearch" not in globals():
            print("INFO: Defining a placeholder for BaseEpsilonGreedyMultiReservoirHPSearch as it was not imported.")
            class BaseEpsilonGreedyMultiReservoirHPSearch:
                def __init__(self, *args, **kwargs): self.best_params_list = None; self.best_score = -np.inf; self.best_reservoir_model = None
                def search(self): print("Placeholder search called.")
                def _get_aggregated_reservoir_outputs_and_targets(self, X, Y, reservoirs, warmup): return np.array([]), np.array([])
        # Create dummy data files if they don't exist for testing basic script flow
        if not os.path.exists("./old_data"): raise FileNotFoundError("Directory './old_data' does not exist.")
    
    n_rc_layers = int(sys.argv[1])
    ASSET_NAME = sys.argv[2]
    print(f"Number of RC layers: {n_rc_layers}")
    print(f"Using data asset: {ASSET_NAME}")

    results = {} 

    X_list_train_raw_full, y_train_targets_full, X_list_test_raw, y_test_targets, original_features, max_timesteps_train_padding = \
        load_FD_data_as_sequences(get_data_file_path(ASSET_NAME))
    
    if X_list_train_raw_full is None or not X_list_train_raw_full :
        print("Failed to load data or full training data is empty. Exiting.")
        exit()
    
    results["original_input_dim"] = original_features
    results["max_timesteps_train_data_for_padding"] = max_timesteps_train_padding

    valid_indices_full_train = [
        i for i, seq in enumerate(X_list_train_raw_full)
        if seq.ndim == 2 and seq.shape[0] > 0 and seq.shape[1] == original_features
    ]
    if not valid_indices_full_train:
        print(f"No valid sequences in full training data for asset {ASSET_NAME} to fit scaler. Exiting.")
        sys.exit()
    
    X_list_train_raw_full_valid = [X_list_train_raw_full[i] for i in valid_indices_full_train]
    y_train_targets_full_aligned = y_train_targets_full[valid_indices_full_train]
    
    print(f"Full training data: {len(X_list_train_raw_full_valid)} valid sequences, {len(y_train_targets_full_aligned)} targets.")

    if len(X_list_train_raw_full_valid) < 2 : 
        print("Not enough valid full training samples to create a validation split. Using all for HPS training, HPS validation will be empty.")
        X_hps_train_raw = list(X_list_train_raw_full_valid) 
        y_hps_train = np.copy(y_train_targets_full_aligned)
        X_hps_val_raw = []
        y_hps_val = np.array([])
    elif VALIDATION_SPLIT_SIZE > 0 and VALIDATION_SPLIT_SIZE < 1:
        hps_train_indices, hps_val_indices = train_test_split(
            np.arange(len(X_list_train_raw_full_valid)),
            test_size=VALIDATION_SPLIT_SIZE,
            random_state=GLOBAL_PARAMS["seed"],
        )
        X_hps_train_raw = [X_list_train_raw_full_valid[i] for i in hps_train_indices]
        y_hps_train = y_train_targets_full_aligned[hps_train_indices]
        X_hps_val_raw = [X_list_train_raw_full_valid[i] for i in hps_val_indices]
        y_hps_val = y_train_targets_full_aligned[hps_val_indices]
        print(f"HPS training data: {len(X_hps_train_raw)} sequences. HPS validation data: {len(X_hps_val_raw)} sequences.")
    else: 
        print("VALIDATION_SPLIT_SIZE is 0 or invalid. Using all valid training data for HPS training. HPS validation will be empty.")
        X_hps_train_raw = list(X_list_train_raw_full_valid)
        y_hps_train = np.copy(y_train_targets_full_aligned)
        X_hps_val_raw = []
        y_hps_val = np.array([])

    print("\n--- Preparing Scaler (fit on FULL valid training data) & Scaling Data Splits ---")
    all_train_timesteps_for_scaler_fit = np.vstack(X_list_train_raw_full_valid)
    scaler_x = StandardScaler().fit(all_train_timesteps_for_scaler_fit)

    X_list_train_scaled_full, _ = scale_data_list(X_list_train_raw_full_valid, scaler_x, original_features, ASSET_NAME, "Full Train")
    X_hps_train_scaled, _       = scale_data_list(X_hps_train_raw, scaler_x, original_features, ASSET_NAME, "HPS Train")
    X_hps_val_scaled, _         = scale_data_list(X_hps_val_raw, scaler_x, original_features, ASSET_NAME, "HPS Val")
    
    X_list_test_scaled, X_list_test_raw_valid = [], []
    y_test_targets_aligned = np.array([]) # Initialize
    if X_list_test_raw:
        valid_indices_test = [
            i for i, seq in enumerate(X_list_test_raw)
            if seq.ndim == 2 and seq.shape[0] > 0 and seq.shape[1] == original_features
        ]
        X_list_test_raw_valid = [X_list_test_raw[i] for i in valid_indices_test]
        if y_test_targets.size > 0 and len(valid_indices_test) <= len(y_test_targets): # ensure indices are valid
            y_test_targets_aligned = y_test_targets[valid_indices_test]
        else:
            y_test_targets_aligned = np.array([]) # if no valid test targets or mismatch
        
        X_list_test_scaled, _ = scale_data_list(X_list_test_raw_valid, scaler_x, original_features, ASSET_NAME, "Test")
    else:
        print("No raw test data to process.")


    print("\n--- PyTorch Convolutional Autoencoder (CAE) Training (on FULL scaled training data) ---")
    if max_timesteps_train_padding == 0 or original_features == 0:
        print(f"Max timesteps for padding ({max_timesteps_train_padding}) or original features ({original_features}) is 0. Cannot train CAE. Exiting.")
        exit()
    if not X_list_train_scaled_full: 
        print("No scaled full training data available for CAE. Exiting.")
        exit()
        
    X_train_full_scaled_padded_torch = pad_list_of_numpy_arrays_to_tensor(
        X_list_train_scaled_full, max_timesteps_train_padding, original_features, CAE_PADDING_VALUE, DEVICE
    )
    X_train_cae_input = X_train_full_scaled_padded_torch.permute(0, 2, 1)

    cae_model_pytorch = CAE1D_PyTorch(
        input_features=original_features, latent_dim=CAE_LATENT_DIM,
        conv_filters=CAE_CONV_FILTERS, kernel_size=CAE_KERNEL_SIZE,
        activation_fn=CAE_ACTIVATION_FN
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cae_model_pytorch.parameters(), lr=1e-3)
    
    cae_dataset = TensorDataset(X_train_cae_input, X_train_cae_input)
    cae_dataloader = DataLoader(cae_dataset, batch_size=CAE_BATCH_SIZE, shuffle=True)

    start_cae_training_time = time.time()
    cae_model_pytorch.train()
    for epoch in range(CAE_EPOCHS):
        epoch_loss = 0
        for batch_inputs, batch_targets in cae_dataloader:
            optimizer.zero_grad()
            outputs = cae_model_pytorch(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == CAE_EPOCHS -1 : # Print less frequently
             print(f"CAE Epoch [{epoch+1}/{CAE_EPOCHS}], Loss: {epoch_loss/len(cae_dataloader):.6f}")
    end_cae_training_time = time.time()
    results["cae_training_time"] = end_cae_training_time - start_cae_training_time
    results["start_cae_training_time"] = start_cae_training_time
    results["end_cae_training_time"] = end_cae_training_time
    results["cae_latent_dim"] = CAE_LATENT_DIM
    print(f"CAE training time: {results['cae_training_time']:.2f}s")

    cae_encoder_pytorch = cae_model_pytorch.encoder
    cae_encoder_pytorch.eval()
    
    # Calculate and store CAE MACs
    cae_macs = count_cae_macs(cae_model_pytorch, max_timesteps_train_padding, original_features)
    results["cae_macs"] = cae_macs
    print(f"CAE MACs: {cae_macs}")


    print("\n--- Encoding Data Splits with Trained PyTorch CAE Encoder ---")
    X_list_train_encoded_full = encode_data_list(X_list_train_scaled_full, cae_encoder_pytorch, max_timesteps_train_padding, original_features, ASSET_NAME, "Full Train")
    X_hps_train_encoded       = encode_data_list(X_hps_train_scaled, cae_encoder_pytorch, max_timesteps_train_padding, original_features, ASSET_NAME, "HPS Train")
    X_hps_val_encoded         = encode_data_list(X_hps_val_scaled, cae_encoder_pytorch, max_timesteps_train_padding, original_features, ASSET_NAME, "HPS Val")
    X_list_test_encoded       = encode_data_list(X_list_test_scaled, cae_encoder_pytorch, max_timesteps_train_padding, original_features, ASSET_NAME, "Test")

    if not X_hps_train_encoded:
        print("HPS training data is empty after encoding. Cannot proceed with HPS. Exiting.")
        sys.exit()
    if not X_hps_val_encoded and VALIDATION_SPLIT_SIZE > 0: 
        print("Warning: HPS validation data is empty after encoding. HPS score will be based on empty set if not handled by HPS class.")
        
    print("\n--- Initializing Hyperparameter Search for RC (on Latent Space) ---")
    hp_search_optimizer = PredictionHPSearch(
        input_dim=CAE_LATENT_DIM,
        X_train_list_encoded=X_hps_train_encoded, 
        y_train_targets=y_hps_train, 
        X_val_list_encoded=X_hps_val_encoded, 
        y_val_targets=y_hps_val,             
        n_iterations=N_HP_ITERATIONS * n_rc_layers, 
        n_reservoirs=n_rc_layers,
        reservoir_hp_space=SEARCH_SPACE, 
        global_params=GLOBAL_PARAMS,
        epsilon_greedy=0.3
    )

    start_hp_search = time.time()
    hp_search_optimizer.search()
    end_hp_search = time.time()
    results["rc_hp_search_time"] = end_hp_search - start_hp_search
    results["start_hp_search"] = start_hp_search
    results["end_hp_search"] = end_hp_search 

    print("\n--- RC Hyperparameter Search Finished ---")
    if not hp_search_optimizer.best_params_list: 
        print("No best params found from HP search. Exiting."); exit()

    best_rc_hps = hp_search_optimizer.best_params_list
    print(f"Best Score (neg MSE on HPS validation set): {hp_search_optimizer.best_score:.5f}")
    if hp_search_optimizer.best_score != -np.inf:
        print(f"Corresponds to best avg HPS validation MSE/sample: {-hp_search_optimizer.best_score:.5f}")
    for i, params in enumerate(best_rc_hps): print(f"  RC Layer {i+1} Best HPs: {params}")


    print("\n--- Retraining Final Model with Best HPs on FULL Training Data ---")
    warmup_final = GLOBAL_PARAMS.get("warmup", 0)
    reservoir_seed_final = GLOBAL_PARAMS.get("seed", None)

    final_reservoirs_list = []
    current_input_dim_for_layer = CAE_LATENT_DIM
    for i, layer_hp in enumerate(best_rc_hps):
        reservoir_node_type = layer_hp.get("RC_node_type", "ESN")
        _activation = layer_hp["activation"]
        shared_params = dict(
            units=layer_hp["units"], input_dim=current_input_dim_for_layer,
            sr=layer_hp["sr"], lr=layer_hp["lr"],
            input_scaling=layer_hp["input_scaling"],
            rc_connectivity=layer_hp.get("connectivity"),
            activation=_activation, W=uniform(high=1.0, low=-1.0), Win=bernoulli,
            seed=reservoir_seed_final + i if reservoir_seed_final is not None else None
        )
        if reservoir_node_type == "IPESN":
            node = IPReservoir(**shared_params,
                            mu=layer_hp.get("mu", 0.0), 
                            learning_rate=layer_hp.get("learning_rate", 1e-3),
                            epochs=GLOBAL_PARAMS.get("epochs", 1))
        else: 
            node = Reservoir(**shared_params)
        final_reservoirs_list.append(node)
        current_input_dim_for_layer = layer_hp["units"]

    print("Preparing FULL training data for final MLP (using all ENCODED training data and best RCs)...")
    final_mlp_train_inputs, final_mlp_train_targets = hp_search_optimizer._get_aggregated_reservoir_outputs_and_targets(
        X_list_train_encoded_full, y_train_targets_full_aligned, 
        final_reservoirs_list, warmup_final
    )

    if final_mlp_train_inputs is None or final_mlp_train_inputs.shape[0] < 1:
        print("Could not generate aggregated RC outputs from FULL training set for final MLP. Exiting.")
        exit()

    if final_mlp_train_targets.ndim > 1 and final_mlp_train_targets.shape[1] == 1:
        final_mlp_train_targets = final_mlp_train_targets.ravel()

    final_mlp = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                             activation=MLP_ACTIVATION, solver=MLP_SOLVER,
                             max_iter=MLP_MAX_ITER * 2, 
                             early_stopping=MLP_EARLY_STOPPING,
                             n_iter_no_change=MLP_N_ITER_NO_CHANGE,
                             random_state=GLOBAL_PARAMS.get("seed"), alpha=MLP_ALPHA)
    print("Training final MLP readout on FULL training data...")
    start_final_mlp_train = time.time()
    final_mlp.fit(final_mlp_train_inputs, final_mlp_train_targets)
    end_final_mlp_train = time.time()
    results["final_mlp_training_time"] = end_final_mlp_train - start_final_mlp_train
    results["start_final_mlp_train"] = start_final_mlp_train
    results["end_final_mlp_train"] = end_final_mlp_train 
    results["mlp_training_time_hps_best_model"] = hp_search_optimizer.best_reservoir_model[2] if hp_search_optimizer.best_reservoir_model and len(hp_search_optimizer.best_reservoir_model) > 2 else None
    print(f"Final MLP training time: {results['final_mlp_training_time']:.2f}s")

    # Calculate and store MLP MACs
    mlp_macs = count_sklearn_mlp_macs(final_mlp)
    results["mlp_macs"] = mlp_macs
    print(f"Final MLP MACs: {mlp_macs}")

    results["final_mlp_prediction_time_test"] = None 
    if X_list_test_encoded and y_test_targets_aligned.size > 0 :
        print("Preparing TEST data for final MLP evaluation (using encoded test data)...")
        final_mlp_test_inputs, final_mlp_test_actual_targets = hp_search_optimizer._get_aggregated_reservoir_outputs_and_targets(
            X_list_test_encoded, y_test_targets_aligned, 
            final_reservoirs_list, warmup_final 
        )

        if final_mlp_test_inputs is None or final_mlp_test_inputs.shape[0] == 0:
            print("Could not generate aggregated RC outputs from test set for final MLP evaluation.")
            r2_final, rmse_final = None, None
        else:
            if final_mlp_test_actual_targets.ndim > 1 and final_mlp_test_actual_targets.shape[1] == 1:
                final_mlp_test_actual_targets = final_mlp_test_actual_targets.ravel()
            
            print("Predicting with final MLP on test data...")
            start_final_mlp_predict_test = time.time()
            test_predictions = final_mlp.predict(final_mlp_test_inputs) 
            end_final_mlp_predict_test = time.time()
            results["final_mlp_prediction_time_test"] = end_final_mlp_predict_test - start_final_mlp_predict_test
            results["start_final_mlp_predict_test"] = start_final_mlp_predict_test
            results["end_final_mlp_predict_test"] = end_final_mlp_predict_test 
            print(f"Final MLP prediction time on test set: {results['final_mlp_prediction_time_test']:.4f}s")
            
            if len(final_mlp_test_actual_targets) > 0 and len(test_predictions) > 0 and \
               len(final_mlp_test_actual_targets) == len(test_predictions): # Ensure lengths match
                 r2_final = r2_score(final_mlp_test_actual_targets, test_predictions)
                 rmse_final = mean_squared_error(final_mlp_test_actual_targets, test_predictions, squared=False)
                 print("\n--- Final Model Performance on Test Set (Prediction) ---")
                 print(f"  R-squared (R²): {r2_final:.4f}")
                 print(f"  Root Mean Squared Error (RMSE): {rmse_final:.4f}")
                 results.update({"r_squared": r2_final, "rmse": rmse_final})
            else:
                print(f"Not enough data or mismatched lengths in test targets ({len(final_mlp_test_actual_targets)}) or predictions ({len(test_predictions)}) to calculate metrics.")
                results.update({"r_squared": None, "rmse": None})
    else:
        print("Skipping final test evaluation as no valid/sufficient encoded test data or targets were available.")
        results.update({"r_squared": None, "rmse": None})

    
    results["model_name"] = f"CAE_PyTorch_RC_MLP_{ASSET_NAME}_val_split"
    n_params_mlp = sum(c.size for c in final_mlp.coefs_) + sum(i.size for i in final_mlp.intercepts_) if hasattr(final_mlp, 'coefs_') else 0
    results["n_model_parameters_mlp_readout"] = n_params_mlp
    
    cae_total_params = sum(p.numel() for p in cae_model_pytorch.parameters() if p.requires_grad)
    results["n_model_parameters_cae"] = cae_total_params
    
    total_params_approx = n_params_mlp + (cae_total_params if cae_total_params else 0)
    results["n_model_parameters_total_approx"] = total_params_approx 

    results["n_rc_layers"] = n_rc_layers
    results["validation_split_size_for_hps"] = VALIDATION_SPLIT_SIZE
    results["best_hps_validation_score_neg_mse"] = hp_search_optimizer.best_score if hasattr(hp_search_optimizer, 'best_score') else -np.inf
    results["best_hps_params"] = str(best_rc_hps)


    results_path = os.path.join(getResultsFolder(ASSET_NAME), f"{n_rc_layers}_rc_layers", f"{ASSET_NAME}_cae_pytorch_rc_pred_results_val_split.csv") 
    save_csv_results(results, results_path)

    print(f"\n--- CAE-PyTorch-RC Prediction (Aggregated MLP Readout with Validation Split HPS) Task for {ASSET_NAME} Finished ---")