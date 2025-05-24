import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import os
from thop import profile, clever_format # <--- ADD THIS IMPORT

# Assumptions:
# 1. X_train.npy, y_train.npy, X_test.npy, y_test.npy files exist in the current directory.
#    X_train.npy should have a shape like (num_samples, num_features)
#    y_train.npy should have a shape like (num_samples,) or (num_samples, 1)
#    Similar for X_test.npy and y_test.npy.
# 2. The 'model_performance' DataFrame is expected to be used for storing results,
#    similar to the TensorFlow script. It will be initialized here.

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# read motor type from args
import sys
if len(sys.argv) == 2:
    motor = sys.argv[1]
else:
    raise ValueError("Please provide the motor type as a command line argument.")

# --- Data Loading ---
# Load data from .npy files
# Ensure these files are in the same directory as this script, or provide full paths.
try:
    X_train_np = np.load(f"./old_data/x_train_{motor}.npy")
    y_train_np = np.load(f"./old_data/y_train_{motor}.npy")
    X_test_np = np.load(f"./old_data/x_test_{motor}.npy")
    y_test_np = np.load(f"./old_data/y_test_{motor}.npy")
except FileNotFoundError:
    print("Error: Ensure X_train.npy, y_train.npy, X_test.npy, y_test.npy are present.")
    raise FileNotFoundError


# Ensure y_train_np and y_test_np are 2D column vectors
if y_train_np.ndim == 1:
    y_train_np = y_train_np.reshape(-1, 1)
if y_test_np.ndim == 1:
    y_test_np = y_test_np.reshape(-1, 1)

# --- Data Preprocessing ---
# Split training data for validation
X_train_s_np, X_val_np, y_train_s_np, y_val_np = train_test_split(
    X_train_np, y_train_np, test_size=0.1, random_state=42 # random_state for reproducibility
)

# Reshape data for LSTM: (batch_size, seq_len, num_features)
# Keras input_shape was (1, X_train.shape[1]), meaning seq_len = 1
# X_train_s_np.shape[1] is num_features
X_train_reshaped_np = X_train_s_np.reshape(X_train_s_np.shape[0], 1, X_train_s_np.shape[1])
X_val_reshaped_np = X_val_np.reshape(X_val_np.shape[0], 1, X_val_np.shape[1])
X_test_reshaped_np = X_test_np.reshape(X_test_np.shape[0], 1, X_test_np.shape[1])

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train_reshaped_np, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s_np, dtype=torch.float32)
X_val_t = torch.tensor(X_val_reshaped_np, dtype=torch.float32)
y_val_t = torch.tensor(y_val_np, dtype=torch.float32)
X_test_t = torch.tensor(X_test_reshaped_np, dtype=torch.float32)
# y_test_t is not directly used in DataLoader for test set if only predicting X
# but good to have for consistency or if loss is computed on test set later
y_test_tensor_for_metrics = torch.tensor(y_test_np, dtype=torch.float32)


# Create TensorDatasets and DataLoaders
batch_size = 500  # From Keras code

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_t, y_val_t)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Test loader will be used for prediction, y component is not strictly needed here for model.predict
# but can be useful if evaluating loss on test set with the loader.
test_dataset = TensorDataset(X_test_t, y_test_tensor_for_metrics)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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

# --- EarlyStopping Callback (PyTorch equivalent) ---
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def restore_best_weights(self, model):
        if self.verbose:
            self.trace_func(f'Loading best model weights from {self.path}')
        model.load_state_dict(torch.load(self.path, map_location=device))


# --- Model Initialization, Loss, Optimizer ---
input_features = X_train_np.shape[1]
model = LSTMModel(input_features).to(device)
print("\nModel Architecture:")
print(model)

dummy_input = torch.randn(1, 1, input_features).to(device)
macs, params_thop = profile(model, inputs=(dummy_input,), verbose=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Keras model used Adam(learning_rate=0.001)

# --- Callbacks Setup ---
# PyTorch ReduceLROnPlateau scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-7, verbose=True
)
# PyTorch EarlyStopping
# Keras EarlyStopping: monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
early_stopper = EarlyStopping(patience=50, verbose=True, path='best_lstm_pytorch_model.pt')


# --- Training Loop ---
epochs = 1000  # From Keras code

print("\nStarting Training...")
start_train_time = time.time()

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * batch_X.size(0) # Weighted by batch size

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_X_val, batch_y_val in val_loader:
            batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
            outputs_val = model(batch_X_val)
            val_loss = criterion(outputs_val, batch_y_val)
            total_val_loss += val_loss.item() * batch_X_val.size(0) # Weighted by batch size

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1:02d}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.1e}")

    scheduler.step(avg_val_loss)
    early_stopper(avg_val_loss, model)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

end_train_time = time.time()
training_time = end_train_time - start_train_time

# Restore best weights according to EarlyStopping criteria (Keras restore_best_weights=True behavior)
if early_stopper.val_loss_min != np.Inf: # Check if a model was saved
    print(f"Restoring best model weights from epoch with val_loss: {early_stopper.val_loss_min:.6f}")
    early_stopper.restore_best_weights(model)
elif epochs > 0 :
    print("No improvement in validation loss observed. Using last model weights or re-check logic if a checkpoint should exist.")


# --- Prediction ---
print("\nStarting Prediction...")
model.eval()
all_predictions_list = []
start_predict_time = time.time()
with torch.no_grad():
    for batch_X_test, _ in test_loader: # y_test from loader not used here
        batch_X_test = batch_X_test.to(device)
        outputs_test = model(batch_X_test)
        all_predictions_list.append(outputs_test.cpu().numpy())

y_predictions_np = np.concatenate(all_predictions_list)  # Shape (n_samples, 1)
y_predictions_np = y_predictions_np[:, 0]  # Adjust shape to (n_samples,) for metrics

end_predict_time = time.time()
prediction_time = end_predict_time - start_predict_time
total_script_time = end_predict_time - start_train_time


# --- Evaluate Model ---
# y_test_np was loaded at the beginning, ensure it's 1D for sklearn metrics
y_test_np_1d = y_test_np.ravel()

r2 = r2_score(y_test_np_1d, y_predictions_np)
rmse = np.sqrt(mean_squared_error(y_test_np_1d, y_predictions_np))


print("\n--- Performance Metrics ---")
print(f"Model R-squared: {r2:.2%}")
print(f"Model RMSE: {rmse:.4f}")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Prediction Time: {prediction_time:.2f} seconds")
print(f"Total Time (Train + Predict): {total_script_time:.2f} seconds")

# save model to file
model_path = f"./red_ai/gpu/lstm_pytorch_model_{motor}.pt"
# create directory if it doesn't exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# get model size in bytes
model_size = os.path.getsize(model_path)

# Record performance metrics
_columns = []
_columns.append('R2')
_columns.append('RMSE')
_columns.append('Train Time')
_columns.append('Predict Time')
_columns.append('Total Time')
_columns.append('startTrainTime')
_columns.append('endTrainTime')
_columns.append('startPredictTime')
_columns.append('endPredictTime')
_columns.append('Model Size (bytes)')
_columns.append('MACs')
_columns.append('Params')
model_performance = pd.DataFrame(columns=_columns)
model_performance.loc['LSTM_PyTorch'] = [r2, rmse, training_time, prediction_time, total_script_time, start_train_time, end_train_time, start_predict_time, end_predict_time, model_size, macs, params_thop]
model_performance.index.name = 'Model'

model_performance_path = f"./red_ai/gpu/lstm_pytorch_model_performance_{motor}.csv"
# create directory if it doesn't exist
os.makedirs(os.path.dirname(model_performance_path), exist_ok=True)
model_performance.to_csv(model_performance_path, index=False)
print(f"Model performance saved to {model_performance_path}")