import torch

class MemoryModel:
    def __init__(self, draft_config, target_config):
        self.draft_config = draft_config
        self.target_config = target_config

    def calculate_memory_lower_bound(self, batch_size, seq_length, dtype_bytes=4):
        # Parameter memory calculation
        draft_param_mem = self.draft_config['num_parameters'] * dtype_bytes
        target_param_mem = self.target_config['num_parameters'] * dtype_bytes

        # Activation memory calculation (including key/value cache)
        draft_activation = 2 * self.draft_config['hidden_size'] * self.draft_config['num_layers'] * seq_length * batch_size * dtype_bytes
        target_activation = 2 * self.target_config['hidden_size'] * self.target_config['num_layers'] * seq_length * batch_size * dtype_bytes

        # Total memory in GB
        total_bytes = draft_param_mem + target_param_mem + draft_activation + target_activation
        return {
            'total': total_bytes / (1024**3),
            'parameters': (draft_param_mem + target_param_mem) / (1024**3),
            'activations': (draft_activation + target_activation) / (1024**3)
        }