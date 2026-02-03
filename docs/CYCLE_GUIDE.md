# vLLM 학습 프로젝트 - 사이클별 구현 가이드

이 문서는 각 Cycle의 상세 구현 가이드입니다. 
각 Cycle은 **구현 → 테스트 → 벤치마크** 순서로 진행합니다.

---

## Cycle 1: Naive Inference (MVP)

### 목표
HuggingFace의 `generate()` 메서드 없이, `forward()`를 직접 호출하여 토큰을 하나씩 생성합니다.

### 왜 이렇게 하나요?
- `generate()`는 내부에 많은 최적화가 숨겨져 있음
- 직접 구현해야 추후 최적화(KV Cache, Batching 등)를 적용할 수 있음
- 이 단계의 **느린 속도가 "기준선"**이 됨

### 구현할 파일

#### 1. `engine/sampler.py`

**목적**: Logits에서 다음 토큰을 선택하는 로직

**구현할 함수**:
```python
def greedy_sample(logits: torch.Tensor) -> int:
    """
    가장 확률 높은 토큰 선택 (Greedy Decoding)
    
    Args:
        logits: 모델 출력 logits [vocab_size] 또는 [1, vocab_size]
    
    Returns:
        선택된 토큰 ID (int)
    
    구현 힌트:
        - torch.argmax() 사용
        - 차원 주의: logits가 2D면 마지막 차원에서 argmax
    """
    pass

def sample_with_temperature(logits: torch.Tensor, temperature: float) -> int:
    """
    Temperature를 적용한 확률적 샘플링
    
    Args:
        logits: 모델 출력 logits
        temperature: 높을수록 다양한 출력 (1.0이 기본)
    
    Returns:
        샘플링된 토큰 ID
    
    구현 힌트:
        1. logits를 temperature로 나눔
        2. softmax로 확률 분포 생성
        3. torch.multinomial()로 샘플링
    """
    pass
```

#### 2. `engine/inference.py`

**목적**: 토큰 생성 루프 구현

**구현할 함수**:
```python
def generate_naive(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 50,
    temperature: float = 1.0
) -> str:
    """
    KV Cache 없이 매번 전체 시퀀스를 처리하는 Naive 추론
    
    Args:
        model: HuggingFace 모델
        tokenizer: 토크나이저
        prompt: 입력 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도
    
    Returns:
        생성된 전체 텍스트
    
    동작 원리:
        1. 프롬프트를 토큰화
        2. 루프 시작:
           a. 전체 token_ids를 모델에 입력 (비효율적!)
           b. 마지막 토큰의 logits 추출
           c. 샘플링으로 다음 토큰 선택
           d. token_ids에 추가
           e. EOS면 종료
        3. 전체 token_ids를 디코딩하여 반환
    
    주의사항:
        - use_cache=False 사용 (또는 기본값)
        - 매 스텝마다 전체 시퀀스가 모델을 통과함
        - 이것이 왜 비효율적인지 이해하는 것이 핵심!
    """
    pass
```

### 테스트 방법

`test/test_naive_inference.py` 생성:
```python
def test_naive_inference():
    """
    테스트 항목:
    1. 텍스트가 정상적으로 생성되는가?
    2. EOS 토큰에서 멈추는가?
    3. max_new_tokens를 초과하지 않는가?
    """
    pass

def test_compare_with_hf_generate():
    """
    HuggingFace generate()와 결과 비교 (greedy 설정 시 동일해야 함)
    
    주의: temperature=0에 가까운 값 사용하여 결정적 출력 확인
    """
    pass
```

### 벤치마크 항목

```python
# benchmarks/benchmark_cycle1.py
def benchmark_naive():
    """
    측정 항목:
    1. 총 생성 시간 (초)
    2. 토큰당 생성 시간 (ms/token)
    3. Throughput (tokens/sec)
    
    테스트 조건:
    - 프롬프트: "Hello, I am a language model"
    - max_new_tokens: 50, 100, 200
    - 3회 반복 평균
    """
    pass
```

### 체크리스트
- [ ] `engine/sampler.py` - `greedy_sample()` 구현
- [ ] `engine/sampler.py` - `sample_with_temperature()` 구현
- [ ] `engine/inference.py` - `generate_naive()` 구현
- [ ] 테스트 통과
- [ ] 벤치마크 기준선 기록

---

## Cycle 2: KV Cache 적용

### 목표
HuggingFace 모델의 `past_key_values`를 활용하여 이미 계산된 K, V를 재사용합니다.

### 핵심 개념

```
[Naive - Cycle 1]
Step 1: model(["Hello"]) → logits
Step 2: model(["Hello", ","]) → logits  ← "Hello" 다시 계산!
Step 3: model(["Hello", ",", "I"]) → logits  ← "Hello", "," 다시 계산!

[KV Cache - Cycle 2]
Step 1 (Prefill): model(["Hello"]) → logits + KV_cache
Step 2 (Decode): model([","], KV_cache) → logits + KV_cache'  ← "," 만 계산!
Step 3 (Decode): model(["I"], KV_cache') → logits + KV_cache''
```

### 구현할 파일

#### `engine/inference.py` 수정

**추가할 함수**:
```python
def generate_with_kv_cache(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 50,
    temperature: float = 1.0
) -> str:
    """
    KV Cache를 활용한 효율적인 추론
    
    Args:
        (generate_naive와 동일)
    
    Returns:
        생성된 전체 텍스트
    
    동작 원리:
        1. 프롬프트 토큰화
        2. Prefill: 전체 프롬프트를 한 번에 처리
           - outputs = model(input_ids, use_cache=True)
           - past_key_values = outputs.past_key_values
        3. Decode 루프:
           - outputs = model(last_token_only, past_key_values=past_key_values, use_cache=True)
           - past_key_values 업데이트
           - 다음 토큰 샘플링
        4. 결과 반환
    
    핵심 차이점:
        - Prefill 후에는 마지막 토큰만 모델에 입력
        - past_key_values를 계속 전달하여 이전 계산 재사용
    """
    pass
```

### past_key_values 이해하기

```python
# 디버깅용 코드 - 이해를 위해 실행해보세요
def inspect_past_key_values(past_key_values):
    """past_key_values 구조 확인"""
    print(f"Number of layers: {len(past_key_values)}")
    print(f"Each layer contains: {len(past_key_values[0])} tensors (key, value)")
    print(f"Key shape: {past_key_values[0][0].shape}")
    print(f"Value shape: {past_key_values[0][1].shape}")
    
    # Shape 의미:
    # [batch_size, num_kv_heads, seq_len, head_dim]
    # seq_len이 토큰 추가될 때마다 증가하는 것 확인!
```

### 테스트 방법

```python
def test_kv_cache_correctness():
    """
    KV Cache 사용 결과가 Naive와 동일한지 확인
    (greedy decoding 시 완전히 동일해야 함)
    """
    pass

def test_past_key_values_growth():
    """
    past_key_values의 seq_len이 토큰 생성마다 1씩 증가하는지 확인
    """
    pass
```

### 벤치마크 항목

```python
def benchmark_kv_cache():
    """
    Cycle 1 대비 성능 비교
    
    측정 항목:
    1. Speedup 비율 (naive_time / kv_cache_time)
    2. 시퀀스 길이별 성능 차이 (짧은 vs 긴 프롬프트)
    
    기대 결과:
    - 시퀀스가 길수록 speedup이 큼
    - 2-5x 속도 향상 예상
    """
    pass
```

### 체크리스트
- [ ] `generate_with_kv_cache()` 구현
- [ ] past_key_values 구조 이해 (inspect 함수 실행)
- [ ] Naive와 결과 동일성 테스트 통과
- [ ] 벤치마크: Cycle 1 대비 speedup 기록

---

## Cycle 3: PagedAttention 기본

### 목표
KV Cache를 고정 크기 블록으로 나누어 관리하는 시스템을 구현합니다.

### 핵심 개념

```
[기존 KV Cache]
Sequence A: [KKKKKKKKKKKK____________] 연속 메모리, 최대 길이 미리 할당

[PagedAttention]
Block Pool: [B0][B1][B2][B3][B4][B5][B6][B7]...

Sequence A (12 tokens, block_size=4):
  block_table = [B0, B3, B5]  ← 비연속 블록 사용
  B0: [K0,K1,K2,K3]
  B3: [K4,K5,K6,K7]
  B5: [K8,K9,K10,K11]
```

### 구현할 파일

#### 1. `engine/sequence.py`

```python
from enum import Enum, auto
from dataclasses import dataclass

class SequenceStatus(Enum):
    """시퀀스 상태"""
    WAITING = auto()   # 대기 중 (아직 시작 안 함)
    RUNNING = auto()   # 실행 중 (토큰 생성 중)
    FINISHED = auto()  # 완료

@dataclass
class SamplingParams:
    """샘플링 파라미터"""
    temperature: float = 1.0
    max_tokens: int = 64
    # ignore_eos: bool = False  # 선택적

class Sequence:
    """
    하나의 생성 요청을 나타내는 클래스
    
    Attributes:
        seq_id: 고유 식별자
        status: 현재 상태
        token_ids: 토큰 ID 리스트 (프롬프트 + 생성된 토큰)
        num_prompt_tokens: 프롬프트 토큰 수
        block_table: 할당된 블록 ID 리스트
        num_cached_tokens: 캐시된 토큰 수 (Prefix Cache용)
        sampling_params: 샘플링 설정
    
    구현할 메서드/프로퍼티:
        - num_tokens: 현재 총 토큰 수
        - num_blocks: 필요한 블록 수
        - num_completion_tokens: 생성된 토큰 수
        - append_token(): 새 토큰 추가
    """
    block_size: int = 256  # 클래스 변수
    
    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = None):
        pass
    
    @property
    def num_tokens(self) -> int:
        """현재 총 토큰 수"""
        pass
    
    @property
    def num_blocks(self) -> int:
        """
        필요한 블록 수 계산
        힌트: (num_tokens + block_size - 1) // block_size
        """
        pass
    
    @property
    def num_completion_tokens(self) -> int:
        """생성된 토큰 수 = 전체 - 프롬프트"""
        pass
    
    def append_token(self, token_id: int):
        """새 토큰 추가"""
        pass
```

#### 2. `engine/block_manager.py`

```python
from collections import deque

class Block:
    """
    KV Cache의 한 블록
    
    Attributes:
        block_id: 블록 고유 ID
        ref_count: 참조 카운트 (이 블록을 사용 중인 시퀀스 수)
    """
    def __init__(self, block_id: int):
        pass
    
    def reset(self):
        """블록 초기화 (재사용 시)"""
        pass

class BlockManager:
    """
    KV Cache 블록 할당/해제 관리자
    
    Attributes:
        block_size: 블록당 토큰 수
        blocks: 모든 블록 리스트
        free_block_ids: 사용 가능한 블록 ID 큐
        used_block_ids: 사용 중인 블록 ID 집합
    
    구현할 메서드:
        - can_allocate(seq): 시퀀스에 필요한 블록 할당 가능?
        - allocate(seq): 시퀀스에 블록 할당
        - deallocate(seq): 시퀀스 블록 반환
        - can_append(seq): 새 토큰 위한 공간 있음?
    """
    
    def __init__(self, num_blocks: int, block_size: int):
        """
        Args:
            num_blocks: 총 블록 수 (GPU 메모리에 따라 결정)
            block_size: 블록당 토큰 수 (보통 256)
        """
        pass
    
    def can_allocate(self, seq: 'Sequence') -> bool:
        """
        시퀀스에 필요한 블록을 할당할 수 있는가?
        
        힌트: free_block_ids 개수 >= seq.num_blocks
        """
        pass
    
    def allocate(self, seq: 'Sequence'):
        """
        시퀀스에 블록 할당
        
        동작:
        1. seq.num_blocks 만큼 free_block_ids에서 가져옴
        2. 각 블록의 ref_count 설정
        3. seq.block_table에 블록 ID 추가
        """
        pass
    
    def deallocate(self, seq: 'Sequence'):
        """
        시퀀스의 블록 반환
        
        동작:
        1. seq.block_table의 각 블록 ref_count 감소
        2. ref_count가 0이면 free_block_ids로 반환
        3. seq.block_table 비우기
        """
        pass
    
    def can_append(self, seq: 'Sequence') -> bool:
        """
        새 토큰을 위한 공간이 있는가?
        
        힌트:
        - 현재 블록에 여유 있으면 True
        - 새 블록 필요하면 free_block_ids 확인
        """
        pass
```

### 테스트 방법

```python
def test_sequence_basic():
    """Sequence 클래스 기본 동작 테스트"""
    seq = Sequence([1, 2, 3, 4, 5], SamplingParams())
    assert seq.num_tokens == 5
    assert seq.num_prompt_tokens == 5
    assert seq.num_completion_tokens == 0
    
    seq.append_token(6)
    assert seq.num_tokens == 6
    assert seq.num_completion_tokens == 1

def test_block_manager_allocate():
    """BlockManager 할당/해제 테스트"""
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = Sequence([1, 2, 3, 4, 5], SamplingParams())  # 5 tokens = 2 blocks
    
    assert bm.can_allocate(seq)
    bm.allocate(seq)
    assert len(seq.block_table) == 2
    assert len(bm.free_block_ids) == 8
    
    bm.deallocate(seq)
    assert len(bm.free_block_ids) == 10
```

### 벤치마크 항목

```python
def benchmark_memory_utilization():
    """
    메모리 효율성 측정
    
    시나리오:
    - 다양한 길이의 시퀀스 100개 생성
    - 총 할당 메모리 vs 실제 사용 메모리 비교
    
    기대: PagedAttention이 연속 할당 대비 메모리 효율 높음
    """
    pass
```

### 체크리스트
- [ ] `Sequence` 클래스 구현
- [ ] `Block` 클래스 구현
- [ ] `BlockManager` 클래스 구현
- [ ] 할당/해제 테스트 통과
- [ ] 메모리 효율성 벤치마크

---

## Cycle 4: Continuous Batching

### 목표
여러 요청을 동시에 처리하되, 완료된 요청은 즉시 빠지고 새 요청은 즉시 들어오는 동적 배치 시스템을 구현합니다.

### 핵심 개념

Scheduler가 매 스텝마다:
1. `waiting` 큐에서 새 요청 가져오기 (Prefill)
2. `running` 큐의 요청 계속 처리 (Decode)
3. 완료된 요청 제거

### 구현할 파일

#### 1. `engine/scheduler.py`

```python
from collections import deque
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager

class Scheduler:
    """
    배치 스케줄러
    
    Attributes:
        block_manager: 블록 관리자
        max_num_seqs: 동시 처리 가능한 최대 시퀀스 수
        waiting: 대기 중인 시퀀스 큐
        running: 실행 중인 시퀀스 큐
    
    구현할 메서드:
        - add(seq): 새 요청 추가
        - schedule(): 다음 스텝 배치 구성
        - postprocess(): 생성 결과 처리, 완료 확인
        - is_finished(): 모든 요청 완료 확인
    """
    
    def __init__(self, block_manager: BlockManager, max_num_seqs: int = 256):
        pass
    
    def add(self, seq: Sequence):
        """새 요청을 waiting 큐에 추가"""
        pass
    
    def is_finished(self) -> bool:
        """모든 요청이 완료되었는가?"""
        pass
    
    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        다음 스텝에 처리할 시퀀스 선택
        
        Returns:
            (sequences, is_prefill): 처리할 시퀀스들, Prefill 여부
        
        로직:
        1. waiting에 요청이 있고 블록 할당 가능하면:
           - Prefill 배치 구성
           - 블록 할당
           - waiting → running 이동
           - return (seqs, True)
        
        2. 그렇지 않으면:
           - running에서 Decode 배치 구성
           - return (seqs, False)
        """
        pass
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], eos_token_id: int):
        """
        생성된 토큰 처리
        
        동작:
        1. 각 시퀀스에 토큰 추가
        2. 종료 조건 확인 (EOS 또는 max_tokens)
        3. 완료된 시퀀스: 블록 해제, running에서 제거
        """
        pass
```

#### 2. `engine/model_runner.py`

```python
class ModelRunner:
    """
    배치 단위 모델 실행기
    
    역할:
    - 시퀀스들을 모델 입력 형태로 변환
    - Prefill/Decode 구분하여 처리
    - 결과에서 다음 토큰 샘플링
    """
    
    def __init__(self, model, tokenizer, device: str):
        pass
    
    def prepare_prefill(self, seqs: list[Sequence]) -> tuple:
        """
        Prefill 입력 준비
        
        - 각 시퀀스의 전체 토큰을 하나의 배치로
        - 패딩 처리 필요 (길이가 다를 수 있음)
        
        Returns:
            (input_ids, attention_mask)
        """
        pass
    
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """
        Decode 입력 준비
        
        - 각 시퀀스의 마지막 토큰만 추출
        
        Returns:
            input_ids: [batch_size, 1]
        """
        pass
    
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        배치 실행 및 다음 토큰 반환
        
        동작:
        1. is_prefill에 따라 입력 준비
        2. 모델 forward
        3. 각 시퀀스의 logits에서 샘플링
        4. 토큰 ID 리스트 반환
        """
        pass
```

### 메인 루프 예시

```python
def run_continuous_batching(scheduler, model_runner, prompts, tokenizer):
    """
    Continuous Batching 메인 루프
    
    사용 예시를 참고하여 전체 흐름 이해
    """
    # 1. 모든 요청을 waiting 큐에 추가
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        seq = Sequence(tokens, SamplingParams(max_tokens=50))
        scheduler.add(seq)
    
    results = {}
    
    # 2. 모든 요청 완료까지 반복
    while not scheduler.is_finished():
        # 스케줄링
        seqs, is_prefill = scheduler.schedule()
        
        if not seqs:
            break
        
        # 모델 실행
        next_tokens = model_runner.run(seqs, is_prefill)
        
        # 후처리
        scheduler.postprocess(seqs, next_tokens, tokenizer.eos_token_id)
        
        # 완료된 시퀀스 결과 저장
        for seq in seqs:
            if seq.status == SequenceStatus.FINISHED:
                results[seq.seq_id] = tokenizer.decode(seq.token_ids)
    
    return results
```

### 테스트 방법

```python
def test_continuous_batching_basic():
    """여러 요청이 동시에 처리되는지 확인"""
    pass

def test_short_request_finishes_first():
    """짧은 요청이 먼저 완료되고 빠지는지 확인"""
    pass
```

### 벤치마크 항목

```python
def benchmark_throughput():
    """
    Throughput 측정
    
    시나리오:
    - 100개 요청 동시 처리
    - 다양한 입력/출력 길이
    
    측정:
    - 전체 처리 시간
    - 총 생성 토큰 수 / 시간 = Throughput
    
    비교:
    - 순차 처리 대비 몇 배 빨라졌는가?
    """
    pass
```

### 체크리스트
- [ ] `Scheduler` 클래스 구현
- [ ] `ModelRunner` 클래스 구현
- [ ] 메인 루프 구현
- [ ] 동시 처리 테스트 통과
- [ ] Throughput 벤치마크

---

## Cycle 5: Prefix Cache

### 목표
동일한 프롬프트의 KV Cache를 해시로 저장하여 재사용합니다.

### 구현할 파일

#### `cache/prefix_cache.py`

```python
import xxhash  # pip install xxhash

class PrefixCache:
    """
    프롬프트 KV Cache 해시 캐싱
    
    동작:
    1. 프롬프트 토큰 → 블록 단위로 해시 계산
    2. 해시로 기존 블록 검색
    3. 히트 시 블록 재사용 (ref_count 증가)
    
    구현할 메서드:
        - compute_hash(): 토큰 블록의 해시 계산
        - lookup(): 캐시에서 매칭 블록 검색
        - store(): 새 블록 캐시에 저장
    """
    
    def __init__(self):
        self.hash_to_block_id: dict[int, int] = {}
    
    @staticmethod
    def compute_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
        """
        토큰 블록의 해시 계산
        
        Args:
            token_ids: 토큰 리스트 (한 블록 분량)
            prefix_hash: 이전 블록의 해시 (체인 해싱)
        
        Returns:
            해시 값
        
        힌트:
            h = xxhash.xxh64()
            if prefix_hash != -1:
                h.update(prefix_hash.to_bytes(8, "little"))
            h.update(bytes(token_ids))  # 또는 numpy 사용
            return h.intdigest()
        """
        pass
    
    def lookup(self, token_ids: list[int], block_size: int) -> tuple[list[int], int]:
        """
        캐시에서 매칭되는 블록 검색
        
        Returns:
            (cached_block_ids, num_cached_tokens)
        """
        pass
```

### BlockManager 수정

```python
# block_manager.py에 추가
def allocate_with_cache(self, seq: Sequence, prefix_cache: PrefixCache):
    """
    Prefix Cache를 활용한 블록 할당
    
    동작:
    1. prefix_cache.lookup()으로 캐시 히트 확인
    2. 히트된 블록은 ref_count만 증가 (재계산 불필요)
    3. 나머지 블록만 새로 할당
    4. seq.num_cached_tokens 업데이트
    """
    pass
```

### 테스트 방법

```python
def test_prefix_cache_hit():
    """동일 프롬프트 연속 요청 시 캐시 히트"""
    pass

def test_partial_cache_hit():
    """부분적으로 같은 프롬프트 (시스템 메시지 공유)"""
    pass
```

### 벤치마크 항목

```python
def benchmark_ttft():
    """
    TTFT (Time To First Token) 측정
    
    시나리오:
    - 동일 시스템 프롬프트 + 다른 유저 메시지
    - 첫 요청 vs 이후 요청의 TTFT 비교
    
    기대: 이후 요청에서 TTFT 50-80% 감소
    """
    pass
```

### 체크리스트
- [ ] `PrefixCache` 클래스 구현
- [ ] `BlockManager.allocate_with_cache()` 구현
- [ ] 캐시 히트 테스트 통과
- [ ] TTFT 벤치마크

---

## Cycle 6: Flash Attention 최적화

### 목표
flash_attn 라이브러리를 통합하여 Attention 연산을 최적화합니다.

### 사전 요구사항
```bash
# CUDA 11.8+ 필요
pip install flash-attn --no-build-isolation
```

### 구현할 파일

#### `layers/attention.py`

```python
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

class FlashAttention:
    """
    Flash Attention 래퍼
    
    주요 함수:
    - flash_attn_varlen_func: 가변 길이 Prefill용
    - flash_attn_with_kvcache: KV Cache Decode용
    """
    
    def prefill_attention(
        self, 
        q, k, v, 
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        block_table=None
    ):
        """
        Prefill 단계 Attention
        
        cu_seqlens: Cumulative sequence lengths
        예: 3개 시퀀스 길이 [5, 3, 7] → cu_seqlens = [0, 5, 8, 15]
        """
        pass
    
    def decode_attention(
        self,
        q,
        k_cache, v_cache,
        context_lens,
        block_table
    ):
        """Decode 단계 Attention"""
        pass
```

### 테스트 방법

```python
def test_flash_attention_correctness():
    """Flash Attention 결과가 기존 구현과 일치하는지 확인"""
    pass
```

### 벤치마크 항목

```python
def benchmark_flash_attention():
    """
    Flash Attention 성능 비교
    
    측정:
    - Throughput 변화
    - GPU 메모리 사용량 변화
    
    기대: 1.5-2x 속도 향상
    """
    pass
```

### 체크리스트
- [ ] flash_attn 설치
- [ ] `FlashAttention` 클래스 구현
- [ ] ModelRunner에 통합
- [ ] 정확성 테스트 통과
- [ ] 성능 벤치마크

---

## Cycle 7: Quantization

### 목표
모델 가중치를 FP16에서 Int8로 양자화하여 메모리를 절감합니다.

### 구현할 파일

#### `quantization/quantizer.py`

```python
import torch

class AbsMaxQuantizer:
    """
    AbsMax 방식 Int8 양자화
    
    원리:
    1. 최대 절대값으로 스케일 계산: scale = max(|weight|) / 127
    2. 양자화: int8 = round(weight / scale)
    3. 역양자화: weight ≈ int8 * scale
    """
    
    @staticmethod
    def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        FP16 → Int8 양자화
        
        Args:
            weight: 원본 가중치
        
        Returns:
            (quantized_weight, scale)
        
        힌트:
            scale = weight.abs().max(dim=-1, keepdim=True).values / 127
            quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
        """
        pass
    
    @staticmethod
    def dequantize(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Int8 → FP16 역양자화"""
        pass


def quantize_model(model) -> dict:
    """
    모델의 모든 Linear 레이어 양자화
    
    동작:
    1. named_modules()로 Linear 레이어 순회
    2. 각 weight 양자화
    3. 양자화된 가중치와 스케일 저장
    """
    pass


def measure_memory(model) -> float:
    """모델 메모리 사용량 측정 (MB)"""
    pass
```

### 테스트 방법

```python
def test_quantization_reconstruction():
    """양자화 → 역양자화 후 오차 확인"""
    pass

def test_model_output_quality():
    """양자화 후 출력 품질 (perplexity) 확인"""
    pass
```

### 벤치마크 항목

```python
def benchmark_quantization():
    """
    양자화 효과 측정
    
    측정:
    - 메모리 사용량 변화
    - 추론 속도 변화 (역양자화 오버헤드)
    - 출력 품질 변화
    
    기대:
    - 메모리 40-50% 절감
    - 품질 저하 5% 미만
    """
    pass
```

### 체크리스트
- [ ] `AbsMaxQuantizer` 구현
- [ ] `quantize_model()` 구현
- [ ] 복원 오차 테스트 통과
- [ ] 품질 테스트 통과
- [ ] 메모리 절감 벤치마크

---

## 최종 정리

### 전체 벤치마크 비교표 작성

각 Cycle 완료 후 `benchmarks/results.md`에 기록:

```markdown
| Cycle | 구현 내용 | Latency | Throughput | Memory | 비고 |
|-------|----------|---------|------------|--------|------|
| 1 | Naive | 10s | 5 tok/s | 1.2GB | 기준선 |
| 2 | KV Cache | 4s | 12.5 tok/s | 1.4GB | 2.5x↑ |
| ... | ... | ... | ... | ... | ... |
```

### 학습 완료 후 다음 단계

- Tensor Parallelism 도전
- Speculative Decoding 구현
- 다른 모델 (Llama, Mistral) 적용
- 실제 vLLM 코드 분석
