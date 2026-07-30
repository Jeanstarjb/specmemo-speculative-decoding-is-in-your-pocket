from pydantic import BaseModel
from typing import List

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

class GenerationRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    max_speculative_steps: int = 5
    early_stop: bool = True

class GenerationResponse(BaseModel):
    generated_texts: List[str]