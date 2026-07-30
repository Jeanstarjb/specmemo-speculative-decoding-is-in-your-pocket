import pytest
import torch
from backend.speculative_decoding import SpeculativeDecoder, MemoryManager

class TestMemoryManager:
    def test_buffer_reuse(self):
        manager = MemoryManager()
        buf1 = manager.get_buffer((10, 20), torch.float32)
        manager.release_buffer(buf1)
        buf2 = manager.get_buffer((10, 20), torch.float32)
        assert buf1.data_ptr() == buf2.data_ptr()

    def test_different_shapes(self):
        manager = MemoryManager()
        buf1 = manager.get_buffer((5, 5), torch.float16)
        buf2 = manager.get_buffer((10, 10), torch.float16)
        assert buf1.shape == (5, 5)
        assert buf2.shape == (10, 10)

class TestSpeculativeDecoderMemory:
    @pytest.fixture
    def decoder(self, draft_model, target_model):
        return SpeculativeDecoder(
            draft_model,
            target_model,
            max_draft_tokens=3,
            buffer_margin=1.5
        )

    def test_memory_reuse(self, decoder):
        initial_buffers = len(decoder.mem_manager.buffers)
        input_ids = torch.tensor([[1, 2, 3]])
        
        # First generation
        output1 = decoder.generate(input_ids)
        # Second generation should reuse buffers
        output2 = decoder.generate(input_ids)
        
        assert output1.shape == output2.shape
        assert len(decoder.mem_manager.buffers) == initial_buffers

    def test_buffer_expansion(self, decoder):
        large_input = torch.tensor([[1] * 1000])
        output = decoder.generate(large_input)
        assert output.shape[1] > 1000
        
        # Verify buffer expansion
        buffer_key = (tuple(int(1000 * 1.5) for _ in large_input.shape), torch.long)
        assert buffer_key in decoder.mem_manager.buffers