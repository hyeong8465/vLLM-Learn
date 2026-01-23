import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.logger import logger


class ModelLoader:
    """
    Singleton 패턴을 사용하는 모델 로더.
    Hugging Face에서 모델과 토크나이저를 다운로드하고 로드합니다.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("ModelLoader instance created")
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Singleton이므로 한 번만 초기화
        if self._initialized:
            return
        
        self.device = self._detect_device()
        logger.info(f"Detected device: {self.device}")
        
        # 프로젝트 루트의 models_cache/ 디렉토리를 캐시로 사용
        project_root = Path(__file__).parent.parent
        self.cache_dir = project_root / "models_cache"
        self.cache_dir.mkdir(exist_ok=True)
        logger.debug(f"Cache directory: {self.cache_dir}")
        
        self.model = None
        self.tokenizer = None
        self.current_model_id = None
        
        self._initialized = True

    def _detect_device(self) -> str:
        """
        사용 가능한 최적의 디바이스를 자동으로 감지합니다.
        우선순위: CUDA > MPS > CPU
        """
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("MPS (Apple Silicon) available")
        else:
            device = "cpu"
            logger.info("Using CPU (no GPU acceleration available)")
        
        return device

    def load_model(self, model_id: str = "Qwen/Qwen3-0.6B") -> tuple:
        """
        지정된 model_id의 모델과 토크나이저를 로드합니다.
        
        Args:
            model_id: Hugging Face 모델 ID (기본값: Qwen/Qwen3-0.6B)
        
        Returns:
            (model, tokenizer) 튜플
    
        Raises:
            RuntimeError: 모델 로드 실패 시
        """
        try:
            # 이미 같은 모델이 로드되어 있으면 재사용
            if self.model is not None and self.current_model_id == model_id:
                logger.info(f"Model '{model_id}' already loaded, reusing existing instance")
                return self.model, self.tokenizer
            
            logger.info(f"Loading model: {model_id}")
            
            # 캐시 경로 설정
            model_cache_path = self.cache_dir / model_id.replace("/", "_")
            
            # 캐시에 모델이 있는지 확인
            if model_cache_path.exists():
                logger.info(f"Model found in cache: {model_cache_path}")
                local_path = str(model_cache_path)
            else:
                logger.info(f"Model not in cache, downloading from Hugging Face...")
                local_path = model_id
            
            # 토크나이저 로드
            logger.debug("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            logger.info("Tokenizer loaded successfully")
            
            # 모델 로드
            logger.info(f"Loading model to {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path,
                cache_dir=str(self.cache_dir),
                device_map=self.device if self.device != "cpu" else None,
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
                trust_remote_code=True
            )
            
            # CPU의 경우 명시적으로 디바이스로 이동
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 모델 파라미터 수 계산
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded successfully. Parameters: {param_count:,}")
            
            self.current_model_id = model_id
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_id}': {e}")
            
            # GPU 메모리 부족 시 CPU로 폴백
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.warning(f"GPU memory error, falling back to CPU")
                self.device = "cpu"
                return self._load_on_cpu(model_id)
            
            raise RuntimeError(f"Failed to load model: {e}")

    def _load_on_cpu(self, model_id: str) -> tuple:
        """
        CPU에서 모델을 강제로 로드합니다 (GPU 실패 시 폴백용).
        """
        try:
            logger.info(f"Loading model on CPU: {model_id}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                dtype=torch.float32,
                trust_remote_code=True
            ).to("cpu")
            
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded on CPU. Parameters: {param_count:,}")
            
            self.current_model_id = model_id
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model on CPU: {e}")
            raise RuntimeError(f"Failed to load model on CPU: {e}")

    def get_model(self):
        """로드된 모델을 반환합니다."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self.model

    def get_tokenizer(self):
        """로드된 토크나이저를 반환합니다."""
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer loaded. Call load_model() first.")
        return self.tokenizer
