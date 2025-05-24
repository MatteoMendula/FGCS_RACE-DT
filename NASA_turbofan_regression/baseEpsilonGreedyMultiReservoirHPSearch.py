import numpy as np
import random
from collections import deque
import copy
import traceback
from reservoirpy.utils import verbosity

# --- BaseEpsilonGreedyMultiReservoirHPSearch class (remains unchanged) ---
class BaseEpsilonGreedyMultiReservoirHPSearch:
    def __init__(self, input_dim, X_train, y_train, X_test, y_test,
                n_iterations, n_reservoirs, reservoir_hp_space, global_params,
                epsilon_greedy=0.3):
        self.input_dim = input_dim
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_iterations = n_iterations
        self.n_reservoirs = n_reservoirs
        self.reservoir_hp_space = reservoir_hp_space
        self.global_params = global_params

        self.topk_memory = deque(maxlen=5)
        self.history = deque(maxlen=n_iterations)
        self.epsilon_greedy = epsilon_greedy

        self.best_score = -np.inf
        self.best_params_list = None
        # self.best_rls_tensor = None # Not used with custom RLS in this way
        self.best_reservoir_model = None # Will store (reservoir_node, rls_weights)
        self.best_reservoir_output_dim = None
        # self.best_all_states_np = None # Not directly stored from evaluate anymore
        # self.best_all_targets_np = None
        # self.best_all_domain_labels_np = None

        self.only_ip_params = ["mu", "learning_rate", "connectivity"] # connectivity is for ESN, not IPESN specific

    def _log_uniform_sample(self, param_range):
        return np.exp(np.random.uniform(param_range["min"], param_range["max"]))

    def _perturb_log_param(self, base_value, param_range, noise_factor=0.1):
        log_base = np.log(base_value)
        log_min = param_range["min"]
        log_max = param_range["max"]
        log_noise = np.random.normal(0, noise_factor * (log_max - log_min))
        perturbed_log = np.clip(log_base + log_noise, log_min, log_max)
        return np.exp(perturbed_log)

    def _sample_single_reservoir_params(self, bias_small_units=True):
        params = {}
        space = self.reservoir_hp_space
        available_units = space["units"]
        if bias_small_units and available_units:
            weights = [1/(u_val+1e-9) for u_val in available_units]
            weights = np.array(weights) / np.sum(weights)
            params["units"] = np.random.choice(available_units, p=weights)
        elif available_units:
            params["units"] = random.choice(available_units)
        else:
            params["units"] = 50 # Default if space["units"] is not well-defined
            print("Warning: 'units' in hp_space is empty or invalid. Defaulting to 50.")

        if "RC_node_type" in space and space["RC_node_type"]:
             params["RC_node_type"] = random.choice(space["RC_node_type"])
        else:
            params["RC_node_type"] = "ESN" # Default
        
        intrinsic_plasticity = (params["RC_node_type"] == "IPESN")

        for key, value_range in space.items():
            if key == "units" or key == "RC_node_type":
                continue
            if key in self.only_ip_params and not intrinsic_plasticity and key != "connectivity": # Keep connectivity for ESN
                continue
            if key == "activation":
                if value_range: params[key] = random.choice(value_range)
                else: params[key] = 'tanh' # Default
            elif isinstance(value_range, dict) and "min" in value_range and "max" in value_range: # Log-uniform
                params[key] = self._log_uniform_sample(value_range)
            elif isinstance(value_range, list) and len(value_range) == 2 and all(isinstance(i, (int, float)) for i in value_range) and key not in ["rls_forgetting_factor"]: # Uniform float/int range
                params[key] = random.uniform(value_range[0], value_range[1])
            elif isinstance(value_range, list) and value_range: # Categorical list
                params[key] = random.choice(value_range)
        return params

    def _perturb_single_reservoir_params(self, base_params):
        perturbed_params = copy.deepcopy(base_params)
        space = self.reservoir_hp_space
        if "RC_node_type" in space and space["RC_node_type"]: # Perturb RC type sometimes
            if random.random() < 0.1: # 10% chance to change RC type
                 perturbed_params["RC_node_type"] = random.choice(space["RC_node_type"])
        
        intrinsic_plasticity = (perturbed_params.get("RC_node_type") == "IPESN")

        for key, base_value in base_params.items():
            if key not in space or key == "RC_node_type": continue # Skip if key not in space or already handled

            if key in self.only_ip_params and not intrinsic_plasticity and key != "connectivity":
                if key in perturbed_params: del perturbed_params[key] # Remove IP param if not IPESN
                continue
            elif key in self.only_ip_params and intrinsic_plasticity and key not in perturbed_params:
                 pass # Will be added if missing for IPESN later

            value_range = space[key]
            perturb_chance = 0.3 # Chance to perturb any given parameter
            if random.random() < perturb_chance:
                if key == 'units':
                    available_units = space.get(key, [])
                    if available_units: # Bias towards smaller units during perturbation too
                        weights = [1/(u_val+1e-9) for u_val in available_units]
                        weights = np.array(weights) / np.sum(weights)
                        perturbed_params[key] = np.random.choice(available_units, p=weights)
                elif key == 'activation':
                    available_activations = space.get(key, [])
                    if available_activations: perturbed_params[key] = random.choice(available_activations)
                elif isinstance(value_range, list) and len(value_range) == 2 and all(isinstance(i, (int, float)) for i in value_range) and key not in ["rls_forgetting_factor"]:
                    noise = np.random.normal(0, 0.1 * (value_range[1] - value_range[0])) # Noise relative to range
                    new_value = base_value + noise
                    perturbed_params[key] = np.clip(new_value, value_range[0], value_range[1])
                elif isinstance(value_range, dict) and "min" in value_range and "max" in value_range: # Log-uniform param
                    perturbed_params[key] = self._perturb_log_param(base_value, value_range)
                elif isinstance(value_range, list) and value_range: # Categorical
                    perturbed_params[key] = random.choice(value_range)
        
        # Ensure IPESN specific params are present if RC_node_type is IPESN
        if intrinsic_plasticity:
            for ip_param_key in self.only_ip_params:
                if ip_param_key not in perturbed_params and ip_param_key in space and ip_param_key != "connectivity":
                    value_range_ip = space[ip_param_key] # Sample it if missing
                    if isinstance(value_range_ip, dict) and "min" in value_range_ip and "max" in value_range_ip:
                         perturbed_params[ip_param_key] = self._log_uniform_sample(value_range_ip)
                    elif isinstance(value_range_ip, list) and value_range_ip:
                         perturbed_params[ip_param_key] = random.choice(value_range_ip)

        return perturbed_params

    def sample_param_list(self):
        params_list = []
        exploit = len(self.topk_memory) > 0 and random.random() > self.epsilon_greedy
        if exploit:
            print("Exploiting memory...")
            if self.n_reservoirs == 1:
                 base_params_for_single_res = random.choice(list(self.topk_memory))
                 perturbed_params = self._perturb_single_reservoir_params(base_params_for_single_res)
                 params_list.append(perturbed_params)
            else: # Multi-reservoir (not the primary focus here but structure kept)
                base_params_list_from_mem = random.choice(list(self.topk_memory))
                if isinstance(base_params_list_from_mem, list) and len(base_params_list_from_mem) == self.n_reservoirs :
                    for i in range(self.n_reservoirs):
                        perturbed_params = self._perturb_single_reservoir_params(base_params_list_from_mem[i])
                        params_list.append(perturbed_params)
                elif isinstance(base_params_list_from_mem, dict) and self.n_reservoirs == 1: # Handles old memory format
                    perturbed_params = self._perturb_single_reservoir_params(base_params_list_from_mem)
                    params_list.append(perturbed_params)
                else:
                    print(f"Warning: Memory item format mismatch. Expected list of {self.n_reservoirs} dicts or single dict for n_res=1. Exploring instead.")
                    exploit = False # Fallback to exploration
        
        if not exploit or not params_list: # Ensure params_list is populated if exploit failed or was skipped
            print("Exploring new parameters...")
            params_list = [] # Clear if exploit failed to populate
            for _ in range(self.n_reservoirs):
                params = self._sample_single_reservoir_params(bias_small_units=True)
                params_list.append(params)
        return params_list

    def search(self):
        print(f"Starting Epsilon-Greedy Search for {self.n_reservoirs} reservoir(s) over {self.n_iterations} iterations...")
        print(f"Exploitation probability: {1.0 - self.epsilon_greedy:.2f}, Exploration probability: {self.epsilon_greedy:.2f}")
        for i in range(self.n_iterations):
            current_params_list = self.sample_param_list()
            if not current_params_list:
                print(f"Iteration {i+1}/{self.n_iterations}: Failed to sample parameters. Skipping.")
                continue
            
            # For single reservoir, current_params_list will have one dict.
            # We expect evaluate to handle this list.
            try:
                # Evaluate returns: score, rls_tensor (None), full_model (tuple), output_dim, ... (None)
                score, _, full_model_tuple, output_dim, _, _, _ = self.evaluate(
                    current_params_list 
                )
            except NotImplementedError:
                print("ERROR: The 'evaluate' method must be implemented in a subclass.")
                raise
            except Exception as e:
                print(f"ERROR during evaluation in iteration {i+1} with params {current_params_list}: {e}")
                traceback.print_exc()
                score = -np.inf 
                full_model_tuple, output_dim = None, None
            
            self.history.append({"params_list": copy.deepcopy(current_params_list), "score": score}) 

            if score > self.best_score:
                self.best_score = score
                self.best_params_list = copy.deepcopy(current_params_list)
                self.best_reservoir_model = full_model_tuple # This is (reservoir_node, rls_weights)
                self.best_reservoir_output_dim = output_dim
                
                # topk_memory stores the parameter dict (or list of dicts for multi-reservoir)
                if self.n_reservoirs == 1:
                    self.topk_memory.append(copy.deepcopy(current_params_list[0]))
                else:
                    self.topk_memory.append(copy.deepcopy(current_params_list))
                print(f"--- Iteration {i+1}/{self.n_iterations}: New best score! Score = {score:.5f} ---")
            else:
                print(f"Iteration {i+1}/{self.n_iterations}: Score = {score:.5f} (Best: {self.best_score:.5f})")
        
        print("\nSearch finished.")
        if self.best_params_list:
            print(f"Best overall score: {self.best_score:.5f}")
        else:
            print("No successful evaluation completed or best_params_list not set.")

    def evaluate(self, current_params_list):
        raise NotImplementedError("Subclasses must implement the evaluate method.")

