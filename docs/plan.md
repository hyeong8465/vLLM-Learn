# vLLM 핵심 기능 구현 프로젝트

## 진행 상황

- [x] Cycle 1: Naive Inference - sampler.py, inference.py(generate_naive), 벤치마크 기준선 측정
- [ ] Cycle 2: KV Cache - inference.py(generate_with_kv_cache), Prefill/Decode 분리, 속도 비교 ← **현재**
- [ ] Cycle 3: PagedAttention - sequence.py, block_manager.py, 블록 기반 KV Cache 관리
- [ ] Cycle 4: Continuous Batching - scheduler.py, model_runner.py, 동적 배치 처리
- [ ] Cycle 5: Prefix Cache - prefix_cache.py, 프롬프트 KV 재사용, TTFT 측정
- [ ] Cycle 6: Flash Attention - attention.py, flash_attn 통합, 최종 성능 측정
- [ ] Cycle 7: Quantization - quantizer.py, Int8 AbsMax, 메모리 절감 측정

## 프로젝트 목표

vLLM(High-throughput LLM inference engine)의 핵심 최적화 기법들을 학습 목적으로 직접 구현합니다.
**레퍼런스**: [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (~1,200줄의 간결한 구현)

---

## 핵심 개념 사전 지식

### 1. LLM 추론의 두 단계

- **Prefill (프리필)**: 입력 프롬프트의 모든 토큰을 한 번에 처리하여 KV Cache 생성
- **Decode (디코드)**: 이전에 생성된 KV Cache를 재사용하며 토큰을 하나씩 생성

### 2. KV Cache란?

Transformer의 Self-Attention에서 계산된 Key, Value 값을 저장해두는 것.
새 토큰 생성 시 이전 토큰들의 K, V를 다시 계산하지 않아도 됨.

### 3. PagedAttention이란?

KV Cache를 고정 크기 "블록"으로 나누어 관리하는 기법.

- 연속된 메모리가 아닌 블록 단위로 할당/해제
- 메모리 단편화 감소, 효율적인 메모리 사용

---

## 프로젝트 구조

```
vLLM-Learn/
├── models/
│   ├── __init__.py
│   └── loader.py          # [완료] HuggingFace 모델 로더
├── engine/
│   ├── __init__.py
│   ├── inference.py       # Cycle 1-2: 기본 추론 엔진
│   ├── sequence.py        # Cycle 3: 시퀀스 관리
│   ├── block_manager.py   # Cycle 3: 블록 기반 KV Cache
│   ├── scheduler.py       # Cycle 4: 배치 스케줄러
│   └── model_runner.py    # Cycle 4: 모델 실행기
├── cache/
│   ├── __init__.py
│   └── prefix_cache.py    # Cycle 5: 프롬프트 캐싱
├── layers/
│   ├── __init__.py
│   ├── attention.py       # Cycle 6: Flash Attention
│   └── sampler.py         # Cycle 1: 샘플링 로직
├── quantization/
│   ├── __init__.py
│   └── quantizer.py       # Cycle 7: Int8 양자화
├── benchmarks/
│   └── benchmark.py       # 각 사이클 벤치마크
├── test/                  # 각 사이클 테스트
└── main.py               # 실행 엔트리 포인트
```

---

## Cycle 1: Naive Inference (MVP)

### 목표

HuggingFace의 `generate()` 메서드 없이, `forward()` 를 직접 호출하여 토큰을 하나씩 생성하는 가장 기본적인 추론 루프를 구현합니다.

### 왜 이렇게 하나요?

- `generate()`는 내부적으로 많은 최적화가 숨겨져 있음
- 직접 구현해야 추후 최적화(KV Cache, Batching 등)를 적용할 수 있음
- 이 단계의 느린 속도가 "기준선"이 됨

### 구현할 파일

**[engine/sampler.py](engine/sampler.py)** - 다음 토큰 선택 로직

```python
# 가장 단순한 Greedy Sampling
def greedy_sample(logits: torch.Tensor) -> int:
    """logits에서 가장 확률 높은 토큰 ID 반환"""
    return logits.argmax(dim=-1).item()

# Temperature 적용 샘플링
def sample_with_temperature(logits: torch.Tensor, temperature: float) -> int:
    """Temperature로 확률 분포 조절 후 샘플링"""
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
```

**[engine/inference.py](engine/inference.py)** - Naive 추론 루프

```python
def generate_naive(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """
    KV Cache 없이 매번 전체 시퀀스를 모델에 입력하는 비효율적인 방식.
    
    동작 원리:
    1. 프롬프트를 토큰화
    2. 루프: 전체 토큰 → 모델 → logits → 다음 토큰 선택 → 토큰 추가
    3. EOS 또는 max_tokens까지 반복
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    for _ in range(max_new_tokens):
        # 매번 전체 시퀀스를 모델에 입력 (비효율적!)
        outputs = model(input_ids)  # use_cache=False 기본값
        next_logits = outputs.logits[:, -1, :]  # 마지막 토큰의 logits
        next_token = greedy_sample(next_logits)
        
        if next_token == tokenizer.eos_token_id:
            break
        
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
    
    return tokenizer.decode(input_ids[0])
```

### 테스트 항목

- 텍스트 생성이 정상 동작하는가?
- HuggingFace `generate()`와 동일한 결과가 나오는가? (greedy 설정 시)

### 벤치마크 지표

- **Latency**: 전체 생성 시간 (초)
- **Tokens/sec**: 초당 생성 토큰 수
- 이 값이 Cycle 2 이후 개선의 "기준선"

---

## Cycle 2: KV Cache 적용

### 목표

HuggingFace 모델의 `past_key_values`를 활용하여 이미 계산된 Key, Value를 재사용합니다.

### 핵심 개념: Prefill vs Decode

```
[Prefill 단계]
입력: "Hello, I am"
      ↓ 전체 토큰을 한 번에 처리
출력: KV Cache (모든 토큰의 K, V 저장)

[Decode 단계 - 반복]
입력: 마지막 생성 토큰 1개 + KV Cache
      ↓ 1개 토큰만 처리
출력: 다음 토큰 + 업데이트된 KV Cache
```

### 구현할 파일

**[engine/inference.py](engine/inference.py)** - KV Cache 활용 추론

```python
def generate_with_kv_cache(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """
    past_key_values를 활용하여 이전 계산 결과를 재사용.
    
    성능 향상 원리:
    - Naive: 매 스텝마다 O(n²) 연산 (n = 현재 시퀀스 길이)
    - KV Cache: 첫 스텝 O(n²), 이후 O(n) 연산
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    past_key_values = None
    
    for i in range(max_new_tokens):
        if past_key_values is None:
            # Prefill: 전체 프롬프트 처리
            outputs = model(input_ids, use_cache=True)
        else:
            # Decode: 마지막 토큰만 처리
            outputs = model(
                input_ids[:, -1:],  # 마지막 토큰만!
                past_key_values=past_key_values,
                use_cache=True
            )
        
        past_key_values = outputs.past_key_values  # KV Cache 저장
        next_logits = outputs.logits[:, -1, :]
        next_token = greedy_sample(next_logits)
        
        if next_token == tokenizer.eos_token_id:
            break
        
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
    
    return tokenizer.decode(input_ids[0])
```

### past_key_values 구조 이해

```python
# past_key_values 구조 (Qwen3-0.6B 기준)
# Tuple of (num_layers) x Tuple of (key, value)
# key.shape = [batch, num_kv_heads, seq_len, head_dim]
# value.shape = [batch, num_kv_heads, seq_len, head_dim]

# 예시: 28개 레이어, 2개 KV heads, 128 head_dim
past_key_values[0][0].shape  # torch.Size([1, 2, 10, 128]) - Layer 0의 Key
past_key_values[0][1].shape  # torch.Size([1, 2, 10, 128]) - Layer 0의 Value
```

### 테스트 항목

- Naive와 동일한 결과가 나오는가?
- past_key_values의 shape이 예상대로 증가하는가?

### 벤치마크 지표

- **Speedup**: Cycle 1 대비 몇 배 빨라졌는가?
- 기대값: **2-5x** 속도 향상 (시퀀스 길이에 따라 다름)

---

## Cycle 3: PagedAttention 기본

### 목표

KV Cache를 고정 크기 블록으로 나누어 관리하는 시스템을 구현합니다.

### 왜 PagedAttention이 필요한가?

```
[기존 방식의 문제]
요청 A: [################____] 16토큰 사용, 4칸 낭비
요청 B: [########____________] 8토큰 사용, 12칸 낭비
→ 연속 메모리 필요, 최대 길이만큼 미리 할당

[PagedAttention]
요청 A: [Block1][Block2][Block3][Block4]  ← 필요한 만큼만 할당
요청 B: [Block5][Block6]
→ 블록 단위로 동적 할당/해제, 메모리 효율 극대화
```

### 구현할 파일

**[engine/sequence.py](engine/sequence.py)** - 시퀀스 상태 관리

```python
from enum import Enum, auto
from dataclasses import dataclass

class SequenceStatus(Enum):
    WAITING = auto()   # 대기 중
    RUNNING = auto()   # 실행 중
    FINISHED = auto()  # 완료

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64

class Sequence:
    """
    하나의 생성 요청을 나타내는 클래스.
    
    핵심 속성:
    - token_ids: 현재까지의 토큰 리스트
    - block_table: 이 시퀀스가 사용 중인 블록 ID 리스트
    - num_cached_tokens: 이미 KV Cache에 저장된 토큰 수
    """
    block_size: int = 256  # 블록당 토큰 수
    
    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = id(self)
        self.status = SequenceStatus.WAITING
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []  # 할당된 블록 ID들
        self.sampling_params = sampling_params
    
    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)
    
    @property
    def num_blocks(self) -> int:
        """필요한 블록 수 계산"""
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    def append_token(self, token_id: int):
        """새 토큰 추가"""
        self.token_ids.append(token_id)
```

**[engine/block_manager.py](engine/block_manager.py)** - 블록 할당 관리

```python
from collections import deque

class Block:
    """KV Cache의 한 블록을 나타냄"""
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0  # 이 블록을 참조하는 시퀀스 수
    
    def reset(self):
        self.ref_count = 1

class BlockManager:
    """
    KV Cache 블록 할당/해제 관리.
    
    동작 원리:
    1. 초기화 시 num_blocks개의 블록 생성
    2. 시퀀스가 요청하면 free_block_ids에서 블록 할당
    3. 시퀀스 완료 시 블록 반환
    """
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()
    
    def can_allocate(self, seq: Sequence) -> bool:
        """시퀀스에 필요한 블록을 할당할 수 있는가?"""
        return len(self.free_block_ids) >= seq.num_blocks
    
    def allocate(self, seq: Sequence):
        """시퀀스에 블록 할당"""
        for _ in range(seq.num_blocks):
            block_id = self.free_block_ids.popleft()
            self.blocks[block_id].reset()
            self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
    
    def deallocate(self, seq: Sequence):
        """시퀀스의 블록 반환"""
        for block_id in seq.block_table:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self.used_block_ids.remove(block_id)
                self.free_block_ids.append(block_id)
        seq.block_table.clear()
    
    def can_append(self, seq: Sequence) -> bool:
        """새 토큰을 위한 공간이 있는가?"""
        # 현재 블록에 여유가 있거나, 새 블록 할당 가능해야 함
        tokens_in_last_block = seq.num_tokens % self.block_size
        if tokens_in_last_block == 0:  # 새 블록 필요
            return len(self.free_block_ids) >= 1
        return True
```

### 블록 테이블 시각화

```
Sequence A (15 tokens, block_size=8):
token_ids:  [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14]
block_table: [Block_0, Block_1]

KV Cache 메모리:
Block_0: [KV(t0), KV(t1), KV(t2), KV(t3), KV(t4), KV(t5), KV(t6), KV(t7)]
Block_1: [KV(t8), KV(t9), KV(t10), KV(t11), KV(t12), KV(t13), KV(t14), ___]
                                                                        ↑ 여유 공간
```

### 테스트 항목

- 블록 할당/해제가 정상 동작하는가?
- 메모리 누수 없이 블록이 재활용되는가?

### 벤치마크 지표

- **Memory Utilization**: 할당된 메모리 중 실제 사용 비율
- **Block Fragmentation**: 낭비되는 블록 수

---

## Cycle 4: Continuous Batching

### 목표

여러 요청을 동시에 처리하되, 완료된 요청은 즉시 빠지고 새 요청은 즉시 들어오는 동적 배치 시스템을 구현합니다.

### Continuous Batching이란?

```
[Static Batching - 기존 방식]
Batch 1: [Req A, Req B, Req C] → 모두 완료될 때까지 대기
         A 완료... B 완료... C 완료... 
Batch 2: [Req D, Req E, Req F] → 다음 배치 시작

[Continuous Batching - vLLM 방식]
Step 1: [A, B, C] 처리
Step 2: [A, B, C] + D 추가 (C 완료되면 빠짐)
Step 3: [A, B, D] + E 추가
→ GPU 항상 바쁘게 유지, Throughput 극대화
```

### 구현할 파일

**[engine/scheduler.py](engine/scheduler.py)** - 배치 스케줄러

```python
from collections import deque
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager

class Scheduler:
    """
    요청을 스케줄링하여 배치를 구성.
    
    핵심 로직:
    1. waiting 큐: 대기 중인 요청
    2. running 큐: 실행 중인 요청
    3. schedule(): prefill/decode 배치 구성
    """
    def __init__(self, block_manager: BlockManager, max_num_seqs: int):
        self.block_manager = block_manager
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
    
    def add(self, seq: Sequence):
        """새 요청 추가"""
        self.waiting.append(seq)
    
    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        다음 스텝에 처리할 시퀀스들 선택.
        
        Returns:
            (sequences, is_prefill): 처리할 시퀀스들과 Prefill 여부
        """
        scheduled = []
        
        # 1. Prefill 우선: waiting에서 가져오기
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break  # 메모리 부족
            
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(seq)
        
        if scheduled:
            return scheduled, True  # Prefill 배치
        
        # 2. Decode: running에서 가져오기
        for seq in list(self.running):
            if self.block_manager.can_append(seq):
                scheduled.append(seq)
        
        return scheduled, False  # Decode 배치
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], eos_token_id: int):
        """생성된 토큰 처리 및 완료 확인"""
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            
            # 종료 조건 확인
            if token_id == eos_token_id or \
               seq.num_tokens - seq.num_prompt_tokens >= seq.sampling_params.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

**[engine/model_runner.py](engine/model_runner.py)** - 배치 실행기

```python
class ModelRunner:
    """배치 단위로 모델 실행"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def prepare_prefill(self, seqs: list[Sequence]):
        """Prefill 입력 준비: 모든 토큰 포함"""
        # 각 시퀀스의 전체 토큰을 하나의 배치로
        input_ids = [seq.token_ids for seq in seqs]
        # 패딩 처리...
        return padded_input_ids, attention_mask
    
    def prepare_decode(self, seqs: list[Sequence]):
        """Decode 입력 준비: 마지막 토큰만"""
        input_ids = [[seq.token_ids[-1]] for seq in seqs]
        return torch.tensor(input_ids)
    
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """배치 실행 및 다음 토큰 샘플링"""
        if is_prefill:
            input_ids, mask = self.prepare_prefill(seqs)
            outputs = self.model(input_ids, attention_mask=mask, use_cache=True)
        else:
            input_ids = self.prepare_decode(seqs)
            outputs = self.model(input_ids, past_key_values=..., use_cache=True)
        
        # 각 시퀀스의 다음 토큰 샘플링
        logits = outputs.logits[:, -1, :]
        next_tokens = self.sample(logits, seqs)
        return next_tokens
```

### 메인 추론 루프

```python
def run_continuous_batching(scheduler, model_runner, prompts):
    """Continuous Batching 메인 루프"""
    # 1. 모든 요청을 waiting 큐에 추가
    for prompt in prompts:
        seq = Sequence(tokenizer.encode(prompt), SamplingParams())
        scheduler.add(seq)
    
    results = {}
    
    # 2. 모든 요청이 완료될 때까지 반복
    while not scheduler.is_finished():
        # 스케줄링: 이번 스텝에 처리할 시퀀스 선택
        seqs, is_prefill = scheduler.schedule()
        
        # 모델 실행
        next_tokens = model_runner.run(seqs, is_prefill)
        
        # 후처리: 토큰 추가, 완료 확인
        scheduler.postprocess(seqs, next_tokens, tokenizer.eos_token_id)
        
        # 완료된 시퀀스 결과 저장
        for seq in seqs:
            if seq.status == SequenceStatus.FINISHED:
                results[seq.seq_id] = tokenizer.decode(seq.token_ids)
    
    return results
```

### 테스트 항목

- 여러 요청이 동시에 처리되는가?
- 짧은 요청이 먼저 완료되고 빠지는가?
- 긴 요청이 계속 진행되는가?

### 벤치마크 지표

- **Throughput**: 초당 처리 토큰 수 (전체)
- **Latency 분포**: 각 요청의 완료 시간
- 기대값: 단일 요청 대비 **2-4x** throughput 향상

---

## Cycle 5: Prefix Cache

### 목표

동일한 프롬프트(시스템 메시지 등)의 KV Cache를 재사용하여 첫 토큰 생성 시간(TTFT)을 줄입니다.

### Prefix Cache 원리

```
요청 1: "You are a helpful assistant. User: Hello"
        [System Prompt KV 계산] + [User 부분 계산]
        
요청 2: "You are a helpful assistant. User: How are you?"
        [System Prompt KV 재사용!] + [User 부분만 계산]
                 ↑ 이미 캐시됨
```

### 구현할 파일

**[cache/prefix_cache.py](cache/prefix_cache.py)** - 프롬프트 KV 캐싱

```python
import xxhash
from functools import lru_cache

class PrefixCache:
    """
    프롬프트의 KV Cache를 Hash로 저장/검색.
    
    동작:
    1. 프롬프트 토큰 → Hash 계산
    2. Hash로 이미 계산된 KV Cache 검색
    3. 있으면 재사용, 없으면 계산 후 저장
    """
    def __init__(self, max_size: int = 100):
        self.cache: dict[int, list[int]] = {}  # hash → block_ids
        self.max_size = max_size
    
    @staticmethod
    def compute_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
        """토큰 시퀀스의 해시 계산"""
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(bytes(token_ids))
        return h.intdigest()
    
    def lookup(self, token_ids: list[int]) -> tuple[list[int], int]:
        """
        캐시에서 매칭되는 prefix 검색.
        
        Returns:
            (cached_block_ids, num_cached_tokens)
        """
        # 블록 단위로 해시 매칭 시도
        block_size = 256
        matched_blocks = []
        prefix_hash = -1
        
        for i in range(0, len(token_ids), block_size):
            block_tokens = token_ids[i:i+block_size]
            if len(block_tokens) < block_size:
                break  # 불완전한 블록은 캐시하지 않음
            
            h = self.compute_hash(block_tokens, prefix_hash)
            if h in self.cache:
                matched_blocks.append(self.cache[h])
                prefix_hash = h
            else:
                break
        
        num_cached = len(matched_blocks) * block_size
        return matched_blocks, num_cached
    
    def store(self, token_ids: list[int], block_ids: list[int]):
        """계산된 KV Cache 블록을 저장"""
        # 블록 단위로 해시 저장
        ...
```

### BlockManager 수정

```python
# block_manager.py에 추가
def allocate_with_cache(self, seq: Sequence, prefix_cache: PrefixCache):
    """캐시 히트 시 블록 재사용"""
    cached_blocks, num_cached = prefix_cache.lookup(seq.token_ids)
    
    if cached_blocks:
        # 캐시된 블록 재사용 (ref_count 증가)
        for block_id in cached_blocks:
            self.blocks[block_id].ref_count += 1
            seq.block_table.append(block_id)
        seq.num_cached_tokens = num_cached
    
    # 나머지 토큰용 새 블록 할당
    remaining_blocks = seq.num_blocks - len(cached_blocks)
    for _ in range(remaining_blocks):
        block_id = self.free_block_ids.popleft()
        self.blocks[block_id].reset()
        seq.block_table.append(block_id)
```

### 테스트 항목

- 동일 프롬프트 연속 요청 시 캐시 히트 발생하는가?
- 캐시 히트 시 Prefill 토큰 수가 줄어드는가?

### 벤치마크 지표

- **TTFT (Time To First Token)**: 첫 토큰 생성까지 시간
- **Cache Hit Rate**: 캐시 히트 비율
- 기대값: 동일 시스템 프롬프트 시 **50-80%** TTFT 감소

---

## Cycle 6: Flash Attention 최적화

### 목표

flash_attn 라이브러리를 통합하여 Attention 연산을 최적화합니다.

### Flash Attention이란?

- GPU 메모리 계층(SRAM, HBM)을 고려한 최적화된 Attention 구현
- 메모리 접근 패턴 최적화로 2-4x 속도 향상
- 정확히 같은 결과, 더 빠른 속도

### 구현할 파일

**[layers/attention.py](layers/attention.py)** - Flash Attention 래퍼

```python
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

class FlashAttention:
    """
    flash_attn 라이브러리를 활용한 최적화된 Attention.
    
    주요 함수:
    - flash_attn_varlen_func: 가변 길이 시퀀스용 (Prefill)
    - flash_attn_with_kvcache: KV Cache 활용 (Decode)
    """
    
    def prefill_attention(self, q, k, v, cu_seqlens_q, cu_seqlens_k, 
                          max_seqlen_q, max_seqlen_k, block_table=None):
        """
        Prefill 단계 Attention.
        
        Args:
            q, k, v: Query, Key, Value 텐서
            cu_seqlens_*: 각 시퀀스의 누적 길이 (Continuous Batching용)
            block_table: PagedAttention용 블록 매핑
        """
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            block_table=block_table
        )
    
    def decode_attention(self, q, k_cache, v_cache, 
                         context_lens, block_table):
        """
        Decode 단계 Attention.
        
        Args:
            q: Query (batch_size, 1, num_heads, head_dim)
            k_cache, v_cache: 전체 KV Cache
            context_lens: 각 시퀀스의 현재 길이
            block_table: 블록 매핑
        """
        return flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=context_lens,
            block_table=block_table,
            causal=True
        )
```

### 설치 요구사항

```bash
# CUDA 11.8+ 필요
pip install flash-attn --no-build-isolation
```

### 테스트 항목

- Flash Attention 결과가 기존 구현과 일치하는가?
- GPU 메모리 사용량이 감소하는가?

### 벤치마크 지표

- **Throughput**: 최종 토큰 처리량
- **Memory Usage**: GPU 메모리 사용량
- 기대값: Cycle 4 대비 **1.5-2x** 속도 향상

---

## Cycle 7: Quantization

### 목표

모델 가중치를 FP16에서 Int8로 양자화하여 메모리 사용량을 줄입니다.

### AbsMax Quantization 원리

```
원본 가중치 (FP16): [-0.5, 0.3, 0.8, -0.2]
                         ↓
1. 최대 절대값 계산: scale = max(|values|) = 0.8
2. 양자화: int8_values = round(values / scale * 127)
   → [-79, 47, 127, -32]
3. 역양자화: values ≈ int8_values * scale / 127
   → [-0.498, 0.296, 0.8, -0.202]

메모리: FP16 (2바이트) → Int8 (1바이트) = 50% 절감
```

### 구현할 파일

**[quantization/quantizer.py](quantization/quantizer.py)** - 양자화 구현

```python
import torch

class AbsMaxQuantizer:
    """AbsMax 방식의 Int8 양자화"""
    
    @staticmethod
    def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        FP16 가중치를 Int8로 양자화.
        
        Args:
            weight: 원본 가중치 (FP16/FP32)
        
        Returns:
            (quantized_weight, scale): Int8 가중치와 스케일
        """
        # 각 출력 채널별로 스케일 계산
        scale = weight.abs().max(dim=-1, keepdim=True).values / 127
        scale = scale.clamp(min=1e-8)  # 0으로 나누기 방지
        
        # 양자화
        quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
        
        return quantized, scale.squeeze()
    
    @staticmethod
    def dequantize(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Int8 가중치를 FP16으로 역양자화"""
        return quantized.float() * scale.unsqueeze(-1)


def quantize_model(model) -> dict:
    """모델의 모든 Linear 레이어 양자화"""
    quantized_state = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            quantized, scale = AbsMaxQuantizer.quantize(weight)
            quantized_state[f"{name}.weight"] = quantized
            quantized_state[f"{name}.scale"] = scale
            
            # 메모리 절약을 위해 원본 삭제
            module.weight.data = AbsMaxQuantizer.dequantize(quantized, scale)
    
    return quantized_state


def measure_memory(model) -> float:
    """모델 메모리 사용량 측정 (MB)"""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / 1024 / 1024
```

### 테스트 항목

- 양자화 전후 출력 차이가 허용 범위 내인가? (perplexity 측정)
- 메모리 사용량이 실제로 줄어드는가?

### 벤치마크 지표

- **Memory Reduction**: 메모리 절감률
- **Quality Degradation**: 출력 품질 저하 정도 (perplexity 비교)
- 기대값: **40-50%** 메모리 절감, **5% 미만** 품질 저하

---

## 최종 벤치마크 비교표

각 사이클 완료 후 아래 지표를 측정하여 기록합니다:


| Cycle | 구현 내용            | Latency   | Throughput | Memory | 비고         |
| ----- | ---------------- | --------- | ---------- | ------ | ---------- |
| 1     | Naive            | 기준선       | 기준선        | 기준선    |            |
| 2     | KV Cache         | -60%      | +150%      | +20%   | KV 저장 오버헤드 |
| 3     | PagedAttention   | 유지        | 유지         | -30%   | 메모리 효율화    |
| 4     | Continuous Batch | 유지        | +200%      | 유지     | GPU 활용률 증가 |
| 5     | Prefix Cache     | -50% TTFT | 유지         | 유지     | 동일 프롬프트 기준 |
| 6     | Flash Attention  | -30%      | +50%       | -20%   |            |
| 7     | Quantization     | +10%      | 유지         | -50%   | 역양자화 오버헤드  |


---

## Future Work

다음 기능들은 현재 범위에서 제외하며, 추후 확장 가능합니다:

- **Tensor Parallelism**: 다중 GPU 분산 처리
- **Speculative Decoding**: 작은 모델로 추측 후 검증
- **Token Streaming**: 실시간 토큰 출력
- **다양한 Sampling**: Top-k, Top-p, Beam Search
- **RoPE 직접 구현**: Rotary Position Embedding 이해

---

## 참고 자료

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): 본 프로젝트의 주요 레퍼런스
- [vLLM Paper](https://arxiv.org/abs/2309.06180): PagedAttention 논문
- [Flash Attention](https://github.com/Dao-AILab/flash-attention): Flash Attention 구현체

