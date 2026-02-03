# vLLM 학습 프로젝트 - 아키텍처 가이드

## 1. 프로젝트 개요

### 1.1 목표
vLLM(High-throughput LLM inference engine)의 핵심 최적화 기법들을 학습 목적으로 직접 구현합니다.

### 1.2 왜 vLLM인가?
- **문제**: LLM 추론은 메모리와 연산량이 매우 큼
- **해결책**: vLLM은 PagedAttention, Continuous Batching 등으로 이 문제를 해결
- **학습 가치**: 이 기법들을 직접 구현하면 LLM 추론의 핵심을 깊이 이해할 수 있음

### 1.3 레퍼런스
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): ~1,200줄의 간결한 vLLM 구현
- [vLLM Paper](https://arxiv.org/abs/2309.06180): PagedAttention 논문

---

## 2. 핵심 개념

### 2.1 LLM 추론의 두 단계

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Inference Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Prefill 단계]                                              │
│  입력: "Hello, I am a"                                       │
│        ↓ 모든 토큰을 한 번에 처리 (병렬)                      │
│  출력: KV Cache 생성 + 첫 번째 새 토큰                        │
│                                                              │
│  [Decode 단계] - 반복                                        │
│  입력: 마지막 생성 토큰 + KV Cache                           │
│        ↓ 1개 토큰만 처리                                     │
│  출력: 다음 토큰 + KV Cache 업데이트                         │
│        ↓ 반복...                                             │
│  종료: EOS 토큰 또는 max_tokens 도달                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 KV Cache란?

Transformer의 Self-Attention에서 **K**ey와 **V**alue를 저장해두는 캐시입니다.

**왜 필요한가?**
```
Attention(Q, K, V) = softmax(QK^T / √d) × V

토큰 10개 → 다음 토큰 생성 시:
- KV Cache 없음: K, V 11개 전부 다시 계산 (O(n²))
- KV Cache 있음: 새 토큰의 K, V만 계산 후 기존에 추가 (O(n))
```

**구조 (Qwen3-0.6B 기준)**:
```python
# past_key_values 구조
# Tuple[Tuple[Tensor, Tensor], ...] - 레이어 수만큼
# 각 레이어: (key, value)
# key.shape = [batch, num_kv_heads, seq_len, head_dim]
# value.shape = [batch, num_kv_heads, seq_len, head_dim]

past_key_values[0][0].shape  # Layer 0의 Key: [1, 2, 10, 128]
past_key_values[0][1].shape  # Layer 0의 Value: [1, 2, 10, 128]
```

### 2.3 PagedAttention이란?

KV Cache를 고정 크기 **블록(Block)** 단위로 관리하는 기법입니다.

**기존 방식의 문제**:
```
요청 A: [████████████____] 12토큰 사용, 4칸 낭비 (최대 16 할당)
요청 B: [████____________] 4토큰 사용, 12칸 낭비
→ 연속 메모리 필요, 최대 길이만큼 미리 할당 → 메모리 낭비
```

**PagedAttention 해결책**:
```
Block Pool: [B0][B1][B2][B3][B4][B5][B6][B7]...

요청 A (12토큰, block_size=4):
  block_table = [B0, B2, B5]  ← 비연속적 할당 가능!
  
요청 B (4토큰):
  block_table = [B1]

→ 필요한 만큼만 블록 할당, 메모리 효율 극대화
```

### 2.4 Continuous Batching이란?

여러 요청을 **동적으로** 배치 처리하는 기법입니다.

**Static Batching (기존)**:
```
Batch 1: [A, B, C] 시작
         A 완료... B 완료... C 완료... (전부 기다림)
Batch 2: [D, E, F] 시작
→ 짧은 요청이 긴 요청을 기다림 → GPU 유휴 시간 발생
```

**Continuous Batching (vLLM)**:
```
Step 1: [A, B, C] 처리
Step 2: [A, B, C, D] ← D 즉시 추가
Step 3: [A, B, D] ← C 완료되어 빠짐
Step 4: [A, B, D, E] ← E 즉시 추가
→ GPU 항상 바쁘게 유지, Throughput 극대화
```

### 2.5 Prefix Cache란?

동일한 프롬프트(시스템 메시지 등)의 KV Cache를 **해시로 저장**하여 재사용합니다.

```
요청 1: "You are a helpful assistant. User: Hello"
        [System Prompt KV 계산: 1초] + [User 부분 계산: 0.1초]
        
요청 2: "You are a helpful assistant. User: How are you?"
        [System Prompt KV 재사용!: 0초] + [User 부분만 계산: 0.1초]
        
→ 동일 시스템 프롬프트 시 TTFT(첫 토큰 시간) 대폭 감소
```

---

## 3. 프로젝트 구조

```
vLLM-Learn/
├── models/                    # 모델 로딩
│   ├── __init__.py
│   └── loader.py             # [완료] HuggingFace 모델 로더
│
├── engine/                    # 추론 엔진 (핵심)
│   ├── __init__.py
│   ├── inference.py          # Cycle 1-2: 기본 추론 루프
│   ├── sampler.py            # Cycle 1: 샘플링 로직
│   ├── sequence.py           # Cycle 3: 시퀀스 상태 관리
│   ├── block_manager.py      # Cycle 3: 블록 기반 KV Cache
│   ├── scheduler.py          # Cycle 4: 배치 스케줄러
│   └── model_runner.py       # Cycle 4: 배치 실행기
│
├── cache/                     # 캐싱
│   ├── __init__.py
│   └── prefix_cache.py       # Cycle 5: 프롬프트 KV 캐싱
│
├── layers/                    # 커스텀 레이어
│   ├── __init__.py
│   └── attention.py          # Cycle 6: Flash Attention 래퍼
│
├── quantization/              # 양자화
│   ├── __init__.py
│   └── quantizer.py          # Cycle 7: Int8 AbsMax
│
├── benchmarks/                # 벤치마크
│   └── benchmark.py          # 각 Cycle 성능 측정
│
├── test/                      # 테스트
│   ├── test_inference.py
│   └── test_kv_cache_comparison.py
│
├── docs/                      # 문서
│   ├── ARCHITECTURE.md       # (현재 파일)
│   └── CYCLE_GUIDE.md        # 사이클별 구현 가이드
│
├── utils/
│   └── logger.py
│
└── main.py                    # 실행 엔트리 포인트
```

---

## 4. nano-vllm 코드 매핑

우리 프로젝트와 nano-vllm의 파일 대응 관계입니다.

| 우리 프로젝트 | nano-vllm | 설명 |
|--------------|-----------|------|
| `engine/sequence.py` | `nanovllm/engine/sequence.py` | 시퀀스 상태 관리 |
| `engine/block_manager.py` | `nanovllm/engine/block_manager.py` | 블록 할당/해제 |
| `engine/scheduler.py` | `nanovllm/engine/scheduler.py` | 배치 스케줄링 |
| `engine/model_runner.py` | `nanovllm/engine/model_runner.py` | 모델 실행 |
| `layers/attention.py` | `nanovllm/layers/attention.py` | Flash Attention |
| `engine/sampler.py` | `nanovllm/layers/sampler.py` | 샘플링 로직 |

### nano-vllm 주요 파일 요약

**`nanovllm/sampling_params.py`**
```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
```

**`nanovllm/engine/sequence.py`**
- `Sequence` 클래스: 하나의 생성 요청 관리
- `block_table`: 할당된 블록 ID 리스트
- `num_cached_tokens`: Prefix Cache 히트 시 캐시된 토큰 수

**`nanovllm/engine/block_manager.py`**
- `Block` 클래스: KV Cache 블록 하나
- `BlockManager`: 블록 할당/해제, 해시 기반 캐시 검색

**`nanovllm/engine/scheduler.py`**
- `waiting` 큐: 대기 중인 요청
- `running` 큐: 실행 중인 요청
- `schedule()`: Prefill vs Decode 배치 구성

---

## 5. 데이터 흐름

### 5.1 단일 요청 처리 흐름

```
┌──────────────────────────────────────────────────────────────┐
│                      Single Request Flow                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. 입력                                                      │
│     prompt = "Hello, I am"                                   │
│         ↓                                                     │
│  2. 토큰화                                                    │
│     token_ids = [15496, 11, 314, 716]                        │
│         ↓                                                     │
│  3. Prefill                                                   │
│     model(token_ids) → logits + past_key_values              │
│         ↓                                                     │
│  4. Sampling                                                  │
│     next_token = argmax(logits[-1]) = 257  # "a"             │
│         ↓                                                     │
│  5. Decode Loop                                               │
│     while not (eos or max_tokens):                           │
│         model([next_token], past_key_values) → logits        │
│         next_token = sample(logits)                          │
│         ↓                                                     │
│  6. 출력                                                      │
│     generated = "Hello, I am a language model..."            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Continuous Batching 흐름

```
┌──────────────────────────────────────────────────────────────┐
│                  Continuous Batching Flow                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Scheduler                                                    │
│  ┌─────────────┐    ┌─────────────┐                          │
│  │  waiting    │    │   running   │                          │
│  │ [D, E, F]   │    │  [A, B, C]  │                          │
│  └─────────────┘    └─────────────┘                          │
│         │                  │                                  │
│         └────────┬─────────┘                                  │
│                  ↓                                            │
│            schedule()                                         │
│                  ↓                                            │
│  ┌─────────────────────────────────┐                         │
│  │ Batch = [A, B, C] (Decode)      │ ← 이미 실행 중인 것들    │
│  │    또는                          │                         │
│  │ Batch = [D] (Prefill)           │ ← 새로운 요청            │
│  └─────────────────────────────────┘                         │
│                  ↓                                            │
│            ModelRunner.run()                                  │
│                  ↓                                            │
│            postprocess()                                      │
│                  ↓                                            │
│  ┌─────────────────────────────────┐                         │
│  │ 완료된 시퀀스 → 결과 반환        │                         │
│  │ 진행 중 시퀀스 → running 유지    │                         │
│  └─────────────────────────────────┘                         │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 성능 지표

각 Cycle에서 측정할 주요 지표입니다.

| 지표 | 설명 | 측정 방법 |
|-----|------|----------|
| **Latency** | 요청 완료까지 시간 | `time.time()` 차이 |
| **Throughput** | 초당 생성 토큰 수 | `total_tokens / total_time` |
| **TTFT** | 첫 토큰 생성 시간 | Prefill 완료까지 시간 |
| **Memory Usage** | GPU 메모리 사용량 | `torch.cuda.memory_allocated()` |
| **Memory Utilization** | 할당된 메모리 중 실제 사용 비율 | `used_blocks / total_blocks` |

### 기대 성능 개선

| Cycle | 주요 개선 | Latency | Throughput | Memory |
|-------|----------|---------|------------|--------|
| 1 → 2 | KV Cache | -60% | +150% | +20% (KV 저장) |
| 2 → 3 | PagedAttention | - | - | -30% |
| 3 → 4 | Continuous Batch | - | +200% | - |
| 4 → 5 | Prefix Cache | -50% TTFT | - | - |
| 5 → 6 | Flash Attention | -30% | +50% | -20% |
| 6 → 7 | Quantization | +10% | - | -50% |

---

## 7. 개발 환경

### 7.1 요구사항
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (Flash Attention 사용 시)

### 7.2 주요 의존성
```
torch
transformers
accelerate
flash-attn  # Cycle 6
xxhash      # Cycle 5 (Prefix Cache)
```

### 7.3 타겟 모델
- **Qwen/Qwen3-0.6B**: 가볍고 테스트하기 적합
- 추후 다른 HuggingFace 모델로 확장 가능
