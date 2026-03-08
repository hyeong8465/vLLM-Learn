"""
Cycle 2: KV Cache 테스트

테스트 항목:
1. 정확성: KV Cache 결과가 Naive와 동일한가? (greedy 설정 시)
2. 구조 확인: past_key_values shape이 토큰 생성마다 증가하는가?
3. 성능: Naive 대비 speedup 측정
"""

import time
import torch

from models import ModelLoader
from utils.logger import logger
from engine.inference import generate_naive, generate_with_kv_cache

# ─────────────────────────────────────────────
# 공통 설정
# ─────────────────────────────────────────────
TEMPERATURE = 0.01   # 결정적 출력을 위해 매우 낮은 값 (greedy에 가깝게)
TOP_K = 1            # greedy: 항상 가장 확률 높은 토큰 선택
MAX_NEW_TOKENS = 50
PROMPT = "Hello, I am"

# 벤치마크용 긴 프롬프트 (context가 길수록 KV Cache 효과가 큼)
LONG_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "In machine learning, large language models are trained on vast amounts of text data. "
    "These models learn to predict the next token given the previous context. "
)


# ─────────────────────────────────────────────
# 테스트 1: 정확성 검증
# ─────────────────────────────────────────────
def test_correctness(model, tokenizer):
    """
    KV Cache 사용 결과가 Naive와 동일한지 확인.
    top_k=1 (greedy)에서 완전히 동일해야 함.
    """
    logger.info("[Test 1] 정확성 검증: KV Cache vs Naive (greedy)")
    logger.info("-" * 50)

    text_naive = generate_naive(
        model, tokenizer, PROMPT, MAX_NEW_TOKENS,
        temperature=TEMPERATURE, k=TOP_K
    )
    text_kv = generate_with_kv_cache(
        model, tokenizer, PROMPT, MAX_NEW_TOKENS,
        temperature=TEMPERATURE, k=TOP_K
    )

    logger.info(f"  Naive:    '{text_naive}'")
    logger.info(f"  KV Cache: '{text_kv}'")

    if text_naive == text_kv:
        logger.info("  ✅ PASS: 두 결과가 동일합니다!")
    else:
        logger.warning("  ⚠️  WARN: 결과가 다릅니다 (temperature > 0이면 정상일 수 있음)")

    return text_naive == text_kv


# ─────────────────────────────────────────────
# 테스트 2: past_key_values 구조 확인
# ─────────────────────────────────────────────
def test_past_key_values_growth(model, tokenizer):
    """
    past_key_values의 seq_len 차원이 토큰 생성마다 1씩 증가하는지 확인.
    """
    logger.info("[Test 2] past_key_values 구조 및 성장 확인")
    logger.info("-" * 50)

    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(model.device)
    prompt_len = input_ids.shape[1]
    past_key_values = None
    seq_lens = []

    with torch.no_grad():
        for step in range(5):
            if past_key_values is None:
                # Prefill
                outputs = model(input_ids, use_cache=True)
            else:
                # Decode: 마지막 토큰만
                outputs = model(
                    input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )

            past_key_values = outputs.past_key_values
            # Layer 0의 Key shape: [batch, num_kv_heads, seq_len, head_dim]
            seq_len = past_key_values[0][0].shape[2]
            seq_lens.append(seq_len)

            if step == 0:
                # 첫 스텝: 구조 출력
                num_layers = len(past_key_values)
                num_kv_heads = past_key_values[0][0].shape[1]
                head_dim = past_key_values[0][0].shape[3]
                logger.info(f"  레이어 수: {num_layers}")
                logger.info(f"  KV heads: {num_kv_heads}, head_dim: {head_dim}")
                logger.info(f"  Prefill 후 seq_len: {seq_len} (프롬프트 길이={prompt_len})")

            # 다음 토큰 샘플링
            next_logits = outputs.logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    logger.info(f"  seq_len 변화: {seq_lens}")

    # 검증: Prefill 후 seq_len = prompt_len, 이후 1씩 증가
    expected = [prompt_len + i for i in range(5)]
    if seq_lens == expected:
        logger.info(f"  ✅ PASS: seq_len이 예상대로 증가합니다 {seq_lens}")
    else:
        logger.warning(f"  ⚠️  WARN: 예상={expected}, 실제={seq_lens}")

    return seq_lens == expected


# ─────────────────────────────────────────────
# 테스트 3: 성능 벤치마크
# ─────────────────────────────────────────────
def benchmark_speedup(model, tokenizer):
    """
    Naive vs KV Cache 속도 비교.
    프롬프트가 길수록 KV Cache 효과가 큼.
    """
    logger.info("[Test 3] 성능 벤치마크: Speedup 측정")
    logger.info("-" * 50)

    test_cases = [
        ("짧은 프롬프트", PROMPT, 30),
        ("긴 프롬프트", LONG_PROMPT, 30),
    ]

    results = []
    for label, prompt, max_tokens in test_cases:
        logger.info(f"\n  [{label}] max_new_tokens={max_tokens}")

        # Naive
        start = time.perf_counter()
        _ = generate_naive(model, tokenizer, prompt, max_tokens, temperature=TEMPERATURE, k=TOP_K)
        time_naive = time.perf_counter() - start

        # KV Cache
        start = time.perf_counter()
        _ = generate_with_kv_cache(model, tokenizer, prompt, max_tokens, temperature=TEMPERATURE, k=TOP_K)
        time_kv = time.perf_counter() - start

        speedup = time_naive / time_kv if time_kv > 0 else float("inf")
        tok_per_sec_naive = max_tokens / time_naive
        tok_per_sec_kv = max_tokens / time_kv

        logger.info(f"    Naive:    {time_naive:.3f}s  ({tok_per_sec_naive:.1f} tok/s)")
        logger.info(f"    KV Cache: {time_kv:.3f}s  ({tok_per_sec_kv:.1f} tok/s)")
        logger.info(f"    Speedup:  {speedup:.2f}x")

        results.append({
            "label": label,
            "time_naive": time_naive,
            "time_kv": time_kv,
            "speedup": speedup,
        })

    return results


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────
def run_all_tests():
    logger.info("=" * 60)
    logger.info("Cycle 2 KV Cache 테스트 시작")
    logger.info("=" * 60)

    loader = ModelLoader()
    model, tokenizer = loader.load_model()

    logger.info("")
    correctness_ok = test_correctness(model, tokenizer)

    logger.info("")
    structure_ok = test_past_key_values_growth(model, tokenizer)

    logger.info("")
    benchmark_results = benchmark_speedup(model, tokenizer)

    # 최종 요약
    logger.info("")
    logger.info("=" * 60)
    logger.info("테스트 결과 요약")
    logger.info("=" * 60)
    logger.info(f"  정확성 테스트:  {'✅ PASS' if correctness_ok else '⚠️  WARN'}")
    logger.info(f"  구조 확인 테스트: {'✅ PASS' if structure_ok else '⚠️  FAIL'}")
    logger.info("  성능 벤치마크:")
    for r in benchmark_results:
        logger.info(f"    {r['label']}: {r['speedup']:.2f}x speedup")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_all_tests()
