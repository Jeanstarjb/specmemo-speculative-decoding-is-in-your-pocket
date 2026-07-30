import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from collections import defaultdict

class MemoryManager:
    """Manages reusable memory buffers for efficient GPU memory utilization"""
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.buffers = defaultdict(list)
        self.in_use = defaultdict(set)
        self.lock = threading.Lock()

    def get_buffer(self, shape: Tuple[int], dtype: torch.dtype) -> torch.Tensor:
        key = (shape, dtype)
        with self.lock:
            if self.buffers[key]:
                buf = self.buffers[key].pop()
                self.in_use[key].add(buf.data_ptr())
                # Ensure correct shape for dynamic dimensions
                if buf.size() != shape:
                    return torch.empty(shape, dtype=dtype, device=self.device)
                return buf
            else:
                new_buf = torch.empty(shape, dtype=dtype, device=self.device)
                self.in_use[key].add(new_buf.data_ptr())
                return new_buf

    def release_buffer(self, tensor: torch.Tensor):
        key = (tuple(tensor.size()), tensor.dtype)
        with self.lock:
            if tensor.data_ptr() in self.in_use[key]:
                self.buffers[key].append(tensor.detach())
                self.in_use[key].remove(tensor.data_ptr())

class SpeculativeDecoder:
    def __init__(self,
                 draft_model: torch.nn.Module,
                 target_model: torch.nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_seq_length: int = 2048,
                 max_draft_tokens: int = 5,
                 buffer_margin: float = 1.2):
        self.draft_model = draft_model.to(device)
        self.target_model = target_model.to(device)
        self.device = device
        self.max_seq_length = max_seq_length
        self.max_draft_tokens = max_draft_tokens
        self.mem_manager = MemoryManager(device)
        
        # Pre-allocate common buffer sizes with margin
        self._preallocate_buffers(buffer_margin)

    def _preallocate_buffers(self, margin: float):
        """Pre-allocate buffers for common tensor shapes"""
        base_shapes = [
            (1, self.max_draft_tokens),  # Draft tokens
            (1, self.max_seq_length),    # Sequence buffers
            (self.draft_model.config.num_attention_heads, self.max_seq_length, self.draft_model.config.hidden_size // self.draft_model.config.num_attention_heads)  # Attention cache
        ]
        
        for shape in base_shapes:
            expanded_shape = tuple(int(dim * margin) for dim in shape)
            for dtype in [torch.float32, torch.float16]:
                self.mem_manager.buffers[(expanded_shape, dtype)] = [
                    torch.empty(expanded_shape, dtype=dtype, device=self.device)
                    for _ in range(2)  # Pre-allocate 2 buffers per shape/dtype
                ]

    def _get_sequence_buffer(self, batch_size: int, seq_length: int, dtype: torch.dtype):
        return self.mem_manager.get_buffer((batch_size, seq_length), dtype)

    def generate(self, input_ids: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> torch.Tensor:
        batch_size = input_ids.size(0)
        current_seq = input_ids.to(self.device)
        
        # Reuse existing memory for sequence storage
        seq_buffer = self._get_sequence_buffer(batch_size, self.max_seq_length, torch.long)
        seq_buffer[:, :input_ids.size(1)] = input_ids
        seq_length = input_ids.size(1)

        while seq_length < self.max_seq_length:
            # Draft phase with memory reuse
            draft_logits, _ = self.draft_model(
                seq_buffer[:, :seq_length],
                cache=self.mem_manager.get_buffer(
                    (batch_size, self.draft_model.config.num_layers, seq_length, 
                     self.draft_model.config.hidden_size),
                    torch.float16
                )
            )
            
            # Sample draft tokens using memory-efficient operations
            draft_tokens = self._sample_tokens(draft_logits, temperature, top_p)
            
            # Verification phase with shared buffers
            target_logits, _ = self.target_model(
                torch.cat([seq_buffer[:, :seq_length], draft_tokens], dim=1),
                cache=self.mem_manager.get_buffer(
                    (batch_size, self.target_model.config.num_layers, seq_length + self.max_draft_tokens,
                     self.target_model.config.hidden_size),
                    torch.float16
                )
            )
            
            # Token validation with in-place operations
            valid_tokens = self._validate_tokens(
                draft_tokens,
                target_logits[:, :draft_tokens.size(1)],
                temperature
            )
            
            # Update sequence buffer with validated tokens
            new_tokens = valid_tokens[:, :self.max_draft_tokens]
            seq_buffer[:, seq_length:seq_length + new_tokens.size(1)] = new_tokens
            seq_length += new_tokens.size(1)
            
            # Early exit if <eos> token generated
            if (new_tokens == 50256).any():
                break

        # Release all buffers back to memory manager
        self.mem_manager.release_buffer(seq_buffer)
        return seq_buffer[:, :seq_length]

    def _sample_tokens(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Memory-efficient sampling with in-place operations"""
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        sorted_probs, indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs <= top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 1
        filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
        sampled_indices = torch.multinomial(filtered_probs, num_samples=1)
        return indices.gather(-1, sampled_indices)

    def _validate_tokens(self, draft_tokens: torch.Tensor, target_logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Validate tokens with memory reuse"""
        target_probs = F.softmax(target_logits / temperature, dim=-1)
        draft_probs = target_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        
        # In-place random number generation
        rand = torch.empty_like(draft_probs).uniform_()
        accepted = (rand < draft_probs).int()
        
        # Find first rejection index
        first_reject = accepted.argmin(dim=1)
        return draft_tokens[:, :first_reject + 1] if first_reject < draft_tokens.size(1) else draft_tokens