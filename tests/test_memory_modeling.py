import pytest
from backend.memory_modeling import MemoryModel
from backend.schemas import ModelConfig

@pytest.fixture
def test_configs():
    return {
        'draft': ModelConfig(
            num_parameters=1000000,
            num_layers=12,
            hidden_size=768,
            attention_heads=12
        ),
        'target': ModelConfig(
            num_parameters=10000000,
            num_layers=24,
            hidden_size=1024,
            attention_heads=16
        )
    }

def test_memory_calculation(test_configs):
    calculator = MemoryModel(
        test_configs['draft'].dict(),
        test_configs['target'].dict()
    )
    result = calculator.calculate_memory_lower_bound(2, 512)
    assert result['total'] > 0
    assert result['parameters'] + result['activations'] == result['total']
    assert result['total'] < 1.0  # For test config values