from models import ModelLoader
from utils.logger import logger

def simple_inference_test():
    logger.info("Starting simple inference test")
    
    # 모델 로드
    loader = ModelLoader()
    model, tokenizer = loader.load_model()
    
    # 프롬프트
    prompt = "Hello, I am"
    
    logger.info(f"Prompt: '{prompt}'")
    logger.info("Generating text...")
    
    # 토큰화
    inputs = tokenizer(prompt, return_tensors="pt").to(loader.device)
    
    # 생성
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 결과
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logger.info(f"Generated: '{generated_text}'")
    
    return generated_text

if __name__ == "__main__":
    simple_inference_test()