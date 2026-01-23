from models import ModelLoader
from utils.logger import logger


def main():
    """
    ModelLoader 테스트 스크립트
    """
    logger.info("=" * 60)
    logger.info("Starting ModelLoader Test")
    logger.info("=" * 60)
    
    try:
        # ModelLoader 인스턴스 생성 (Singleton)
        loader = ModelLoader()
        logger.info(f"Using device: {loader.device}")
        logger.info(f"Cache directory: {loader.cache_dir}")
        
        # 모델과 토크나이저 로드 (기본값: Qwen/Qwen2.5-0.5B-Instruct)
        logger.info("\n" + "-" * 60)
        logger.info("Loading model and tokenizer...")
        logger.info("-" * 60)
        
        model, tokenizer = loader.load_model()
        
        # 모델 정보 출력
        logger.info("\n" + "=" * 60)
        logger.info("Model Information")
        logger.info("=" * 60)
        logger.info(f"Model ID: {loader.current_model_id}")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Tokenizer type: {type(tokenizer).__name__}")
        logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")
        
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {param_count:,}")
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        # 간단한 토큰화 테스트
        logger.info("\n" + "-" * 60)
        logger.info("Tokenization Test")
        logger.info("-" * 60)
        
        test_text = "Hello, how are you?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        logger.info(f"Input text: '{test_text}'")
        logger.info(f"Tokens: {tokens}")
        logger.info(f"Decoded: '{decoded}'")
        
        logger.info("\n" + "=" * 60)
        logger.info("ModelLoader Test Completed Successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
