import pytest
import torch
from backend.speculative_decoding import SpeculativeDecoder

class MockDraftModel(torch.nn.Module):
    def init_cache(self, batch_size, seq_length):
        return 'draft_cache'
    
    def forward(self, input_ids, cache):
        return torch.ones(input_ids.size(0), 1, 1000) * 0.5, 'new_draft_cache'

class MockTargetModel(torch.nn.Module):
    def init_cache(self, batch_size, seq_length):
        return 'target_cache'
    
    def forward(self, input_ids, cache):
        return torch.ones(input_ids.size(0), 1, 1000) * 0.5, 'new_target_cache'

@pytest.fixture
def decoder():
    return SpeculativeDecoder(
        draft_model=MockDraftModel(),
        target_model=MockTargetModel(),
        max_draft_tokens=3,
        device='cpu'
    )

def test_decoder_initialization(decoder):
    assert decoder.draft_model is not None
    assert decoder.target_model is not None
    assert decoder.device == 'cpu'

def test_generate_candidates(decoder):
    input_ids = torch.tensor([[1, 2, 3]])
    candidates, probs = decoder._generate_candidates(input_ids)
    assert candidates.shape == (1, 3)
    assert probs.shape == (1, 3)

def test_full_acceptance(decoder):
    input_ids = torch.tensor([[1]])
    output = decoder.decode(input_ids, max_tokens=3)
    assert output.shape == (1, 4)

def test_rejection_handling(decoder):
    decoder.target_model.forward = lambda x, _: (torch.randn(1, 4, 1000), None)
    input_ids = torch.tensor([[1]])
    output = decoder.decode(input_ids, max_tokens=1)
    assert output.shape == (1, 2)