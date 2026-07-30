import torch
import torch.nn.functional as F
from typing import Tuple, Optional

class SpeculativeDecoder:
    def __init__(self,
                 draft_model: torch.nn.Module,
                 target_model: torch.nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_seq_length: int = 2048,
                 max_draft_tokens: int = 5):
        self.draft_model = draft_model.to(device)
        self.target_model = target_model.to(device)
        self.device = device
        self.max_seq_length = max_seq_length
        self.max_draft_tokens = max_draft_tokens
        
        # Cache buffers for both models
        self.draft_cache = None
        self.target_cache = None

    def _get_model_outputs(self, model, input_ids, cache):
        if cache is None:
            cache = model.init_cache(input_ids.size(0), self.max_seq_length)
        
        with torch.inference_mode():
            logits, new_cache = model(input_ids, cache)
        return logits[:, -1, :], new_cache

    def _generate_candidates(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        candidates = []
        draft_probs = []
        
        current_ids = input_ids
        for _ in range(self.max_draft_tokens):
            logits, self.draft_cache = self._get_model_outputs(
                self.draft_model, current_ids, self.draft_cache)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            candidates.append(next_token)
            draft_probs.append(probs.gather(-1, next_token))
            current_ids = next_token
        
        return torch.cat(candidates, dim=1), torch.cat(draft_probs, dim=1)

    def _verify_candidates(self, input_ids: torch.Tensor, candidates: torch.Tensor,
                          draft_probs: torch.Tensor) -> Tuple[torch.Tensor, int]:
        combined = torch.cat([input_ids, candidates], dim=1)
        target_logits, self.target_cache = self._get_model_outputs(
            self.target_model, combined, self.target_cache)
        
        target_probs = F.softmax(target_logits[:, :-1], dim=-1)
        adjusted_probs = target_probs / draft_probs
        adjusted_probs = torch.clamp(adjusted_probs, 0, 1)
        
        # Find first rejection point
        uniform = torch.rand_like(adjusted_probs)
        accepted = (uniform < adjusted_probs).int()
        first_reject = torch.argmin(accepted, dim=1)
        
        # Handle all accepted case
        if torch.all(accepted):
            return candidates[:, :self.max_draft_tokens], self.max_draft_tokens
        
        n_accepted = first_reject.min().item()
        if n_accepted == 0:
            # Sample from target distribution
            corrected_token = torch.multinomial(target_probs[:, 0], 1)
            return corrected_token, 0
        
        return candidates[:, :n_accepted], n_accepted

    def decode_step(self, input_ids: torch.Tensor) -> torch.Tensor:
        candidates, draft_probs = self._generate_candidates(input_ids)
        verified_tokens_accepted = self._verify_candidates(input_ids, candidates, draft_probs)
        return verified_tokens

    def decode(self, input_ids: torch.Tensor, max_tokens: int = 50) -> torch.Tensor:
        generated = input_ids.to(self.device)
        for _ in range(max_tokens):
            if generated.size(1) >= self.max_seq_length:
                break
            new_tokens = self.decode_step(generated[:, -1:])
            generated = torch.cat([generated, new_tokens], dim=1)
        return generated