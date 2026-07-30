import torch
import threading
from typing import List, Optional
from .speculative_decoding import SpeculativeDecoder
from transformers import AutoTokenizer

class DistributedSpeculator:
    def __init__(self, draft_model, target_model, num_gpus: Optional[int] = None):
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.decoders = []
        for i in range(self.num_gpus):
            device = f'cuda:{i}'
            decoder = type(draft_model)().to(device)
            decoder.load_state_dict(draft_model.state_dict())
            target = type(target_model)().to(device)
            target.load_state_dict(target_model.state_dict())
            self.decoders.append(SpeculativeDecoder(decoder, target))

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        chunk_size = (len(prompts) + self.num_gpus - 1) // self.num_gpus
        chunks = [prompts[i*chunk_size:(i+1)*chunk_size] for i in range(self.num_gpus)]
        
        results = [[] for _ in range(self.num_gpus)]
        threads = []
        
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            thread = threading.Thread(
                target=self._process_chunk,
                args=(i, chunk, results, kwargs)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        return [text for sublist in results for text in sublist]

    def _process_chunk(self, gpu_id: int, chunk: List[str], results: list, kwargs: dict):
        device = f'cuda:{gpu_id}'
        inputs = self.tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=kwargs.get('max_length', 512),
            return_tensors='pt'
        ).input_ids.to(device)
        
        outputs = self.decoders[gpu_id].generate(
            inputs,
            max_length=kwargs.get('max_tokens', 50) + inputs.size(1),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9)
        )
        results[gpu_id] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)