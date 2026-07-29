from fastapi import FastAPI
from .memory_modeling import MemoryModel
from .schemas import MemoryRequest, MemoryResponse

app = FastAPI()

@app.post('/api/calculate-memory', response_model=MemoryResponse)
async def calculate_memory(req: MemoryRequest):
    calculator = MemoryModel(
        draft_config=req.draft_config.dict(),
        target_config=req.target_config.dict()
    )
    result = calculator.calculate_memory_lower_bound(
        req.batch_size,
        req.sequence_length,
        req.dtype_bytes
    )
    return {
        'memory_lower_bound_gb': result['total'],
        'parameter_memory_gb': result['parameters'],
        'activation_memory_gb': result['activations']
    }

@app.get('/health')
async def health_check():
    return {'status': 'healthy'}