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
    class AENet(nn.Module):
        def __init__(self, input_dim, block_size): # block_size is for the unused cov parameters
            super(AENet, self).__init__()
            self.input_dim = input_dim
            # These covariance matrices are not used in the forward pass as defined in the original model.
            # They are kept here for consistency with the provided AENet structure.
            # If they are not part of the saved state_dict, their initialization here is fine.
            # If they ARE part of the saved state_dict and you don't want to load them,
            # you might need to filter the state_dict before loading.
            self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
            self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
                nn.ReLU(),
                nn.Linear(128, 8), # Bottleneck layer
                nn.BatchNorm1d(8, momentum=0.01, eps=1e-03),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.Linear(8, 128),
                nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
                nn.ReLU(),
                nn.Linear(128, self.input_dim) # Output layer reconstructs input
            )

        def forward(self, x):
            # Input x is expected to be (batch_of_frames, input_dim)
            # The view(-1, self.input_dim) handles cases where x might have an extra sequence dimension
            # that needs to be flattened with the batch dimension for nn.Linear.
            # However, for frame-wise processing, ensure data loader provides (num_frames_in_batch, input_dim).
            x_reshaped = x.view(-1, self.input_dim)
            z = self.encoder(x_reshaped)
            reconstructed_x = self.decoder(z)
            return reconstructed_x, z

    # --- Model Loading Logic ---
    model_filename = 'AE_example_checkpoint.pth'
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
        loaded_model = AENet(input_dim=INPUT_DIM, block_size=BLOCK_SIZE)
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
