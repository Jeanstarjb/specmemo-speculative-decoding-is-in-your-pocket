from pydantic import BaseModel

class ModelConfig(BaseModel):
    num_parameters: int
    num_layers: int
    hidden_size: int
    attention_heads: int

class MemoryRequest(BaseModel):
    draft_config: ModelConfig
    target_config: ModelConfig
    batch_size: int = 1
    sequence_length: int = 256
    dtype_bytes: int = 4

class MemoryResponse(BaseModel):
    memory_lower_bound_gb: float
    parameter_memory_gb: float
    activation_memory_gb: float