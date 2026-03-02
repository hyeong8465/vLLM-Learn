import torch
from typing import Optional

def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1, keepdim=True)

def sample_with_temperature_topk(
    logits: torch.Tensor, 
    temperature: float, k: Optional[int] = None
    ) -> torch.Tensor:
    if k is not None:
        k_val = min(k, logits.size(-1))
        topk_values, _ = torch.topk(logits, k_val, dim=-1)
        threshold = topk_values[..., -1:].expand_as(logits)  # k번째로 큰 값을 threshold로
        logits = torch.where(logits >= threshold, logits, torch.tensor(float('-inf'), device=logits.device))
    
    probs = torch.softmax(logits / temperature, dim=-1)
    
    return torch.multinomial(probs, num_samples=1)
