# vLLM-Learn 성능 측정 보고서

---

## Cycle 1 — Naive Inference vs HuggingFace `generate()`

### 환경

| 항목 | 내용 |
|------|------|
| 디바이스 | Apple Silicon MPS (MacBook Air) |
| 모델 | Qwen/Qwen3-0.6B (596,049,920 parameters) |
| 테스트 파일 | `test/test_basic_inference.py` |
| 엔진 파일 | `engine/naive_inference.py` |

### 테스트 설정

| 항목 | 값 |
|------|----|
| Prompt | `"Hello, I am"` |
| max_new_tokens | 10 |
| temperature | 0.01 |
| top_k | 5 |

---

### 성능 측정 결과

3회 실행 평균 (2026-03-02)

| 실행 | Naive Inference | HuggingFace `generate()` | 배수 |
|------|-----------------|--------------------------|------|
| Run 1 | 2.567s (3.90 tok/s) | 0.760s (13.17 tok/s) | 3.4x |
| Run 2 | 2.281s (4.38 tok/s) | 0.720s (13.88 tok/s) | 3.2x |
| Run 3 | 1.959s (5.10 tok/s) | 0.747s (13.38 tok/s) | 2.6x |
| **평균** | **2.269s (4.46 tok/s)** | **0.742s (13.48 tok/s)** | **~3.1x** |

> HuggingFace `generate()`가 약 **3.1배** 빠름

---

### 성능 차이 이유

#### 1. KV Cache 유무

Naive Inference는 매 스텝마다 **전체 시퀀스를 처음부터** 모델에 입력한다.
토큰이 1개씩 늘어날 때마다 이전 토큰들의 Key·Value 값을 반복해서 계산한다.

```
Step 1: [t1]           → forward() → 1개 토큰 처리
Step 2: [t1, t2]       → forward() → 2개 토큰 처리
Step 3: [t1, t2, t3]   → forward() → 3개 토큰 처리
...
```

시퀀스 길이가 `n`이면 총 연산량은 `O(n²)`에 비례한다.

HuggingFace `generate()`는 내부적으로 `past_key_values`(KV Cache)를 사용한다.
첫 스텝(Prefill)에서만 전체 시퀀스를 처리하고, 이후 스텝(Decode)에서는 새로 추가된 **토큰 1개**만 처리한다.

```
Prefill: [t1, t2, t3, t4] → forward() → KV Cache 생성
Decode:  [t5] + KV Cache  → forward() → 1개 토큰만 처리
Decode:  [t6] + KV Cache  → forward() → 1개 토큰만 처리
...
```

이후 스텝의 연산량은 `O(n)`으로 줄어든다.

#### 2. Python 루프 오버헤드

Naive Inference는 Python `for` 루프 안에서 토큰을 하나씩 처리한다.
각 스텝마다 Python 인터프리터 오버헤드와 새 텐서 생성(`torch.cat`, `torch.tensor`) 비용이 누적된다.

HuggingFace `generate()`는 내부적으로 최적화된 C++/CUDA 커널과 배치 연산을 활용하여 이 오버헤드를 최소화한다.

---

### 요약

Naive Inference는 KV Cache 없이 매 스텝마다 전체 시퀀스를 재계산하기 때문에 느리다.
이 비효율이 Cycle 2(KV Cache 적용)의 개선 목표이며, 이 결과가 이후 사이클의 **기준선(baseline)**이 된다.

---
