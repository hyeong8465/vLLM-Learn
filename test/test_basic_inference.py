import time

from models import ModelLoader
from utils.logger import logger
from engine.inference import generate_naive

Temperature = 1
MaxNewTokens = 10
TopK = 5
PROMPT = "Hello, I am"


def _count_new_tokens(tokenizer, prompt: str, generated_text: str) -> int:
    """생성된 텍스트에서 프롬프트를 제외한 새 토큰 수 계산"""
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    full_len = len(tokenizer.encode(generated_text, add_special_tokens=False))
    return max(0, full_len - prompt_len)


def run_naive_inference(model, tokenizer) -> tuple[str, float]:
    """generate_naive 실행 및 소요 시간 반환"""
    logger.info("Running naive inference (generate_naive)...")
    start = time.perf_counter()
    generated_text = generate_naive(
        model,
        tokenizer,
        PROMPT,
        MaxNewTokens,
        temperature=Temperature,
        k=TopK,
    )
    elapsed = time.perf_counter() - start
    logger.info(f"Generated: '{generated_text}'")
    return generated_text, elapsed


def run_huggingface_inference(model, tokenizer, loader) -> tuple[str, float]:
    """HuggingFace model.generate 실행 및 소요 시간 반환"""
    logger.info("Running HuggingFace inference (model.generate)...")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(loader.device)
    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MaxNewTokens,
        temperature=Temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        top_k=TopK,
    )
    elapsed = time.perf_counter() - start
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logger.info(f"Generated: '{generated_text}'")
    return generated_text, elapsed


def compare_inference_times():
    """두 추론 방식을 실행하고 시간 비교"""
    logger.info("Starting comparison: naive inference vs HuggingFace inference")
    logger.info("=" * 60)

    # 모델 한 번만 로드
    loader = ModelLoader()
    model, tokenizer = loader.load_model()
    logger.info(f"Prompt: '{PROMPT}', max_new_tokens={MaxNewTokens}")
    logger.info("=" * 60)

    # 1. Naive inference
    text_naive, time_naive = run_naive_inference(model, tokenizer)
    num_tokens_naive = _count_new_tokens(tokenizer, PROMPT, text_naive)
    tokens_per_sec_naive = num_tokens_naive / time_naive if time_naive > 0 else 0

    logger.info("=" * 60)

    # 2. HuggingFace inference
    text_hf, time_hf = run_huggingface_inference(model, tokenizer, loader)
    num_tokens_hf = _count_new_tokens(tokenizer, PROMPT, text_hf)
    tokens_per_sec_hf = num_tokens_hf / time_hf if time_hf > 0 else 0

    # 3. 결과 요약
    logger.info("=" * 60)
    logger.info("⏱️  Time Comparison")
    logger.info("-" * 40)
    logger.info(f"  naive inference:  {time_naive:.3f}s ({num_tokens_naive} tokens → {tokens_per_sec_naive:.2f} tokens/sec)")
    logger.info(f"  HuggingFace:      {time_hf:.3f}s ({num_tokens_hf} tokens → {tokens_per_sec_hf:.2f} tokens/sec)")
    logger.info("-" * 40)
    if time_hf > 0:
        ratio = time_naive / time_hf
        logger.info(f"  HuggingFace is {ratio:.1f}x faster")
    logger.info("=" * 60)


if __name__ == "__main__":
    compare_inference_times()