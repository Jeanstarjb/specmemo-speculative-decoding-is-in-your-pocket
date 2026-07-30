from fastapi import FastAPI
import torch
from .memory_modeling import MemoryModel
from .schemas import MemoryRequest, MemoryResponse, GenerationRequest, GenerationResponse
from .speculative_decoding import SpeculativeDecoder
from .distributed import DistributedSpeculator

app = FastAPI()

class DraftModel(torch.nn.Module):
    def init_cache(self, batch_size, seq_length):
        return None
    
    def forward(self, input_ids, cache):
        return torch.randn(input_ids.size(0), 1, 50257), None

class TargetModel(torch.nn.Module):
    def init_cache(self, batch_size, seq_length):
        return None
    
    def forward(self, input_ids, cache):
        return torch.randn(input_ids.size(0), 1, 50257), None

num_gpus = torch.cuda.device_count()
draft_model = DraftModel()
target_model = TargetModel()
spec_decoder = DistributedSpeculator(draft_model, target_model) if num_gpus > 1 \
    else SpeculativeDecoder(draft_model, target_model)

@app.post("/memory-estimate", response_model=MemoryResponse)
async def estimate_memory(request: MemoryRequest):
    calculator = MemoryModel(request.draft_config.dict(), request.target_config.dict())
    mem_info = calculator.calculate_memory_lower_bound(
        request.batch_size,
        request.sequence_length,
        request.dtype_bytes
    )
    return MemoryResponse(**mem_info)

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    generated = spec_decoder.generate(
        request.prompts,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )
    return GenerationResponse(generated_texts=generated)