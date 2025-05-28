# load_and_report.py (super-debug version)
import sys

# Try to write to stderr and stdout immediately and flush
# This helps see if basic I/O is working at the very start.
sys.stderr.write("DEBUG: Script attempting to start via stderr.\n")
sys.stderr.flush()
sys.stdout.write("DEBUG: Script attempting to start via stdout.\n")
sys.stdout.flush()

print("DEBUG: Python print() statement reached in script.", flush=True)

try:
    import os
    print(f"DEBUG: import 'os' successful. Current Working Directory: {os.getcwd()}", flush=True)
    
    # List contents of /app to verify files from Python's perspective
    try:
        app_contents = os.listdir('/app')
        print(f"DEBUG: Contents of /app directory: {app_contents}", flush=True)
    except Exception as e_ls:
        print(f"DEBUG: Error listing /app directory: {e_ls}", flush=True)

    import time
    print("DEBUG: import 'time' successful.", flush=True)

    # Attempt to import joblib and report outcome
    try:
        import joblib
        print("DEBUG: import 'joblib' successful.", flush=True)
    except ImportError as e_joblib:
        print(f"DEBUG: IMPORT ERROR: Failed to import 'joblib'. Error: {e_joblib}", flush=True)
        print("DEBUG: Ensure 'scikit-learn' is listed in your requirements file and installed during Docker build.", flush=True)
        sys.exit(1) # Exit if joblib cannot be imported, crucial dependency

    # --- Model Loading Logic ---
    model_filename = 'untrained_mlp_regressor.joblib'
    # WORKDIR is /app, so files copied with `COPY . .` are in the current dir for the script
    model_path = os.path.join(os.getcwd(), model_filename) 

    print(f"DEBUG: Model path to check: '{model_path}'", flush=True)

    if not os.path.exists(model_path):
        print(f"DEBUG: ERROR: Model file '{model_path}' not found!", flush=True)
        sys.exit(1) # Exit if model file is not found
    
    print(f"DEBUG: Model file '{model_path}' found. Attempting to load...", flush=True)
    
    start_time_load = time.perf_counter()
    loaded_model = joblib.load(model_path)
    end_time_load = time.perf_counter()
    load_duration = end_time_load - start_time_load
    
    print(f"DEBUG: Model '{model_filename}' loaded successfully from '{model_path}'.", flush=True)
    print(f"DEBUG: Time taken to load the model: {load_duration:.6f} seconds.", flush=True)
    print("Container ready (model loaded).") # Final readiness message

except Exception as e_main:
    print(f"DEBUG: An unhandled error occurred in the main try block: {e_main}", flush=True)
    # Print full traceback for detailed debugging
    import traceback
    traceback.print_exc(file=sys.stdout) # Print to stdout
    sys.stdout.flush() # Ensure traceback is flushed
    sys.exit(1) # Exit due to error

print("DEBUG: Script finished execution successfully.", flush=True)
sys.exit(0) # Explicitly exit with success code
