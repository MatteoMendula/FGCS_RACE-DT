# load_torch_model_and_report.py (super-debug version)
import sys
import time
import torch
import torch.nn as nn

# Try to write to stderr and stdout immediately and flush
# This helps see if basic I/O is working at the very start.
sys.stderr.write("DEBUG: Script attempting to start via stderr - ciao.\n")
sys.stderr.flush()
sys.stdout.write("DEBUG: Script attempting to start via stdout.\n")
sys.stdout.flush()

print("DEBUG: Python print() statement reached in script.", flush=True)

try:
    import os
    print(f"DEBUG: import 'os' successful. Current Working Directory: {os.getcwd()}", flush=True)

    # List contents of /app to verify files from Python's perspective
    try:
        # Assuming the script runs in /app or a similar environment where the model file is expected.
        # Adjust this path if your execution environment is different.
        current_dir_contents = os.listdir('.') # list current directory
        print(f"DEBUG: Contents of current directory ('.') : {current_dir_contents}", flush=True)
        if os.path.exists('/app'):
            app_contents = os.listdir('/app')
            print(f"DEBUG: Contents of /app directory: {app_contents}", flush=True)
        else:
            print("DEBUG: /app directory does not exist.", flush=True)
    except Exception as e_ls:
        print(f"DEBUG: Error listing directory: {e_ls}", flush=True)

    import time
    print("DEBUG: import 'time' successful.", flush=True)

    # Attempt to import PyTorch and report outcome
    try:
        import torch
        import torch.nn as nn
        print(f"DEBUG: import 'torch' and 'torch.nn' successful. PyTorch version: {torch.__version__}", flush=True)
    except ImportError as e_torch:
        print(f"DEBUG: IMPORT ERROR: Failed to import 'torch'. Error: {e_torch}", flush=True)
        print("DEBUG: Ensure 'torch' is listed in your requirements file and installed.", flush=True)
        sys.exit(1) # Exit if torch cannot be imported, crucial dependency

    # === DNN Autoencoder Definition ===
    # --- PyTorch LSTM Model Definition ---
    class LSTMModel(nn.Module):
        def __init__(self, input_features):
            super(LSTMModel, self).__init__()
            # Layer 1: Bidirectional LSTM
            self.lstm1 = nn.LSTM(input_size=input_features, hidden_size=128,
                                batch_first=True, bidirectional=True)
            # Keras BatchNormalization default axis is -1.
            # LSTM output is (batch, seq_len, features). For seq_len=1, it's (batch, 1, 2*128)
            # We apply BN on the features dimension (2*128=256)
            self.bn1 = nn.BatchNorm1d(2 * 128)
            self.dropout1 = nn.Dropout(0.3)

            # Layer 2: Bidirectional LSTM
            # Input features for lstm2 is 2*128 (output of lstm1)
            self.lstm2 = nn.LSTM(input_size=2 * 128, hidden_size=64,
                                batch_first=True, bidirectional=True)
            # Output of lstm2 (if return_sequences=False in Keras) is concatenated final hidden states
            # which is (batch, 2*64=128)
            self.bn2 = nn.BatchNorm1d(2 * 64)
            self.dropout2 = nn.Dropout(0.3)

            # Dense Layers
            self.fc1 = nn.Linear(2 * 64, 64)
            self.relu = nn.ReLU()
            self.dropout3 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):
            # x shape: (batch, seq_len=1, features)

            # LSTM Layer 1
            # out1: (batch, seq_len=1, num_directions * hidden_size1=256)
            out1, _ = self.lstm1(x)

            # Batch Normalization 1
            # Permute or squeeze/unsqueeze for BatchNorm1d if seq_len > 1
            # Here seq_len is 1, so squeeze: (batch, 256)
            out1_squeezed = out1.squeeze(1)
            bn1_out = self.bn1(out1_squeezed)
            # UnSqueeze back if needed for next LSTM: (batch, 1, 256)
            bn1_out_unsqueezed = bn1_out.unsqueeze(1)
            drop1_out = self.dropout1(bn1_out_unsqueezed)

            # LSTM Layer 2
            # input: (batch, seq_len=1, 256)
            # Keras return_sequences=False: use final hidden state.
            # For PyTorch BiLSTM, hn is (num_layers*num_directions, batch, hidden_size2=64)
            _, (hn2, _) = self.lstm2(drop1_out)
            # Concatenate final forward (hn2[-2]) and backward (hn2[-1]) hidden states
            # Resulting shape: (batch, num_directions * hidden_size2=128)
            out2_hidden_concat = torch.cat((hn2[-2, :, :], hn2[-1, :, :]), dim=1)

            # Batch Normalization 2
            bn2_out = self.bn2(out2_hidden_concat) # Applied on (batch, 128)
            drop2_out = self.dropout2(bn2_out)

            # Dense Layers
            fc1_out = self.fc1(drop2_out)
            relu_out = self.relu(fc1_out)
            drop3_out = self.dropout3(relu_out)
            output = self.fc2(drop3_out)  # Shape: (batch, 1)
            return output

    # --- Model Loading Logic ---
    model_filename = 'lstm_pytorch_model_FD001.pth'
    # WORKDIR is /app, so files copied with `COPY . .` are in the current dir for the script
    # Or, if running locally, it's the directory where you execute the script.
    model_path = os.path.join(os.getcwd(), model_filename)

    print(f"DEBUG: Model path to check: '{model_path}'", flush=True)

    if not os.path.exists(model_path):
        print(f"DEBUG: ERROR: Model file '{model_path}' not found!", flush=True)
        sys.exit(1) # Exit if model file is not found

    print(f"DEBUG: Model file '{model_path}' found. Attempting to load...", flush=True)

    # IMPORTANT: Define model parameters before loading the state_dict
    # You MUST set these to the values used when the model was trained and saved.
    # For example:
    INPUT_DIM = 13  # Placeholder: Adjust to your model's actual input dimension
    BLOCK_SIZE = 13   # Placeholder: Adjust if your model uses a different block_size for cov matrices

    print(f"DEBUG: Initializing AENet with input_dim={INPUT_DIM}, block_size={BLOCK_SIZE}", flush=True)
    try:
        loaded_model = LSTMModel(input_dim=INPUT_DIM)
        print("DEBUG: AENet model instance created.", flush=True)
    except Exception as e_init:
        print(f"DEBUG: ERROR: Failed to initialize AENet model: {e_init}", flush=True)
        sys.exit(1)

    start_time_load = time.perf_counter()
    try:
        # Load the checkpoint. It might be a dictionary containing the state_dict and other info.
        # Or it might be the state_dict directly.
        # common practice: checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), ...}
        # If it's just the state_dict: state_dict = torch.load(model_path)
        
        # Determine if CUDA is available and set map_location accordingly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEBUG: Using device: {device}", flush=True)

        checkpoint = torch.load(model_path, map_location=device)
        print(f"DEBUG: Raw checkpoint loaded from '{model_path}'. Type: {type(checkpoint)}", flush=True)

        # Assuming the checkpoint is a dictionary and the model's state_dict is stored under a key like 'model_state_dict' or 'state_dict'.
        # If torch.save(model.state_dict(), PATH) was used, then checkpoint is the state_dict itself.
        if isinstance(checkpoint, dict) and ('model_state_dict' in checkpoint or 'state_dict' in checkpoint):
            state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
            print(f"DEBUG: Checkpoint is a dict. Attempting to load state_dict from key: '{state_dict_key}'", flush=True)
            state_dict_to_load = checkpoint[state_dict_key]
        elif isinstance(checkpoint, dict) and not ('model_state_dict' in checkpoint or 'state_dict' in checkpoint):
            # If it's a dict but doesn't have the common keys, it might be the state_dict itself,
            # but with unexpected structure. This branch tries to load it directly.
            # Or it could be that the keys are different, e.g. 'net', 'model'
            # You might need to inspect your .pth file to know the exact structure.
            print(f"DEBUG: Checkpoint is a dict but does not contain typical state_dict keys. Keys found: {list(checkpoint.keys())}. Assuming checkpoint IS the state_dict.", flush=True)
            state_dict_to_load = checkpoint
        else: # If checkpoint is not a dict, assume it's the state_dict directly
            print("DEBUG: Checkpoint is not a dict. Assuming it is the state_dict directly.", flush=True)
            state_dict_to_load = checkpoint
        
        # Before loading, you might want to inspect keys if there are mismatches:
        # print(f"DEBUG: Keys in loaded state_dict: {state_dict_to_load.keys()}")
        # print(f"DEBUG: Keys in model architecture: {loaded_model.state_dict().keys()}")

        loaded_model.load_state_dict(state_dict_to_load)
        print("DEBUG: model.load_state_dict() successful.", flush=True)
        
        loaded_model.eval() # Set the model to evaluation mode
        print("DEBUG: Model set to evaluation mode (model.eval()).", flush=True)

    except Exception as e_load_model:
        print(f"DEBUG: ERROR: Failed to load model state_dict: {e_load_model}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)

    end_time_load = time.perf_counter()
    load_duration = end_time_load - start_time_load

    print(f"DEBUG: PyTorch Model '{model_filename}' loaded successfully from '{model_path}'.", flush=True)
    print(f"DEBUG: Time taken to load the model: {load_duration:.6f} seconds.", flush=True)
    print("Container ready (PyTorch model loaded).") # Final readiness message

except Exception as e_main:
    print(f"DEBUG: An unhandled error occurred in the main try block: {e_main}", flush=True)
    # Print full traceback for detailed debugging
    import traceback
    traceback.print_exc(file=sys.stdout) # Print to stdout
    sys.stdout.flush() # Ensure traceback is flushed
    sys.exit(1) # Exit due to error

print("DEBUG: Script finished execution successfully.", flush=True)
sys.exit(0) # Explicitly exit with success code


