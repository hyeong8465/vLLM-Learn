import torch
from models.loader import ModelLoader
from typing import Optional
from layers.sampler import sample_with_temperature_topk

def generate_naive(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens,
    temperature: float = 1.0,
    k: Optional[int] = None
    ) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)

            logits = outputs.logits
            last_logits = logits[:,-1,:]

            next_token_id = sample_with_temperature_topk(last_logits, temperature, k)
            if next_token_id == tokenizer.eos_token_id:
                break

        input_ids = torch.cat([input_ids, next_token_id], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())


def generate_with_kv_cache(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    temperature: float = 1.0,
    k: Optional[int] = None
    ) -> str:

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    past_key_values = None

    for _ in range(max_new_tokens):
        # inference
        if past_key_values is None: # prefill
            outputs = model(input_ids, use_cache=True)
        else:
            outputs = model(
                input_ids[:,-1:],
                past_key_values=past_key_values,
                use_cache=True
            )

        past_key_values = outputs.past_key_values # kv 캐시 업데이트
        next_logits = outputs.logits[:, -1, :]
        next_token = sample_with_temperature_topk(next_logits, temperature, k)

        if next_token == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
    model, tokenizer = ModelLoader().load_model()
    print(generate_naive(model, tokenizer, "Hello, I am", 10, temperature=0.01, k=5))
    print(generate_with_kv_cache(model, tokenizer, "Hello, I am", 10, temperature=0.01, k=5))