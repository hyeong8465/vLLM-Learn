# vLLM-Learn 성능 측정 보고서

---

## Cycle 2 — KV Cache (`generate_with_kv_cache`) vs Naive (`generate_naive`)

### 환경

| 항목 | 내용 |
|------|------|
| 디바이스 | Apple Silicon MPS (MacBook Air) |
| 모델 | Qwen/Qwen3-0.6B (596,049,920 parameters) |
| 테스트 파일 | `test/test_kv_cache.py` |
| 엔진 파일 | `engine/inference.py` |

### 테스트 설정

| 항목 | 값 |
|------|----|
| max_new_tokens | 30 |
| temperature | 0.01 |
| top_k | 1 (greedy) |
| 반복 횟수 | 3회 |

---

### 성능 측정 결과 — 짧은 프롬프트

**Prompt:** `"Hello, I am"` (4 tokens)

3회 실행 결과 (2026-03-08)

| 실행 | Naive | KV Cache | Speedup |
|------|-------|----------|---------|
| Run 1 | 3.853s (7.79 tok/s) | 1.394s (21.52 tok/s) | 2.76x |
| Run 2 | 2.363s (12.70 tok/s) | 1.178s (25.47 tok/s) | 2.01x |
| Run 3 | 2.375s (12.63 tok/s) | 1.186s (25.30 tok/s) | 2.00x |
| **평균** | **2.864s (11.04 tok/s)** | **1.253s (24.10 tok/s)** | **~2.26x** |

> Run 1이 느린 이유: MPS 첫 실행 시 커널 컴파일 워밍업 비용

---

### 성능 측정 결과 — 긴 프롬프트

**Prompt:** `"The quick brown fox jumps over the lazy dog. In machine learning, ..."` (40 tokens)

3회 실행 결과 (2026-03-08)

| 실행 | Naive | KV Cache | Speedup |
|------|-------|----------|---------|
| Run 1 | 5.423s (5.53 tok/s) | 1.399s (21.44 tok/s) | 3.88x |
| Run 2 | 3.850s (7.79 tok/s) | 1.675s (17.91 tok/s) | 2.30x |
| Run 3 | 3.993s (7.51 tok/s) | 1.451s (20.68 tok/s) | 2.75x |
| **평균** | **4.422s (6.94 tok/s)** | **1.508s (20.01 tok/s)** | **~2.98x** |

---

### 프롬프트 길이별 Speedup 비교

| 프롬프트 | Naive 평균 | KV Cache 평균 | Speedup |
|---------|-----------|--------------|---------|
| 짧은 (4 tokens) | 2.864s | 1.253s | **2.26x** |
| 긴 (40 tokens) | 4.422s | 1.508s | **2.98x** |

> 프롬프트가 길수록 Speedup이 커짐 — KV Cache 이론과 일치

---

### 정확성 검증

greedy 설정(top_k=1, temperature=0.01) 기준으로 Naive와 KV Cache가 **완전히 동일한 텍스트를 생성**함을 확인.

```
Naive:    'Hello, I am a student who is taking a course in the field of mathematics...'
KV Cache: 'Hello, I am a student who is taking a course in the field of mathematics...'
```

---

### KV Cache 구조 확인

Qwen3-0.6B의 `past_key_values` 구조 (Prefill 후):

| 항목 | 값 |
|------|----|
| 레이어 수 | 28 |
| KV heads | 8 |
| head_dim | 128 |
| Prefill 후 seq_len | 프롬프트 길이와 동일 |

Decode 매 스텝마다 seq_len이 정확히 1씩 증가함을 확인: `[4, 5, 6, 7, 8]`

---

### 성능 차이 이유

#### 1. KV Cache로 중복 연산 제거

Naive는 매 스텝마다 이전 토큰들의 Key·Value를 반복해서 계산한다.

```
Step 1: model([t1])           → O(1²) 연산
Step 2: model([t1, t2])       → O(2²) 연산  ← t1 재계산!
Step 3: model([t1, t2, t3])   → O(3²) 연산  ← t1, t2 재계산!
```

KV Cache는 Prefill에서 전체 프롬프트를 한 번만 처리하고, Decode에서는 새 토큰 1개만 처리한다.

```
Prefill: model([t1, t2, ..., tn])    → KV Cache 생성
Decode:  model([t_new] + KV Cache)   → O(n) 연산, 이전 K·V 재사용
```

프롬프트가 길수록 Naive의 누적 연산량이 기하급수적으로 늘어나 Speedup이 더 커진다.

#### 2. MPS 환경의 특성

Apple MPS는 CUDA 대비 매 `model()` 호출마다 고정 오버헤드(커널 컴파일, 동기화)가 크다.
짧은 시퀀스에서는 이 고정 비용이 절감된 연산보다 커져 Speedup이 상대적으로 작게 측정된다.
CUDA GPU + 긴 시퀀스 환경에서는 이론값인 **2~5x**에 더 근접할 것으로 예상된다.

---

### Cycle 1 대비 요약

| 지표 | Cycle 1 Naive | Cycle 2 KV Cache | 개선 |
|------|--------------|-----------------|------|
| 짧은 프롬프트 tok/s | 4.46 | 24.10 | **+441%** |
| 긴 프롬프트 tok/s | 측정 안 함 | 20.01 | — |
| 알고리즘 복잡도 | O(n²) per step | O(n) per step | — |

> Cycle 1 기준선(4.46 tok/s) 대비 Cycle 2 KV Cache(24.10 tok/s)는 약 **5.4x** 향상

---
