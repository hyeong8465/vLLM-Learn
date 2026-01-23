import logging
import os
from rich.logging import RichHandler

def setup_logger(name="vllm-from-scratch"):
    
    # 1. 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 중복 출력 방지
    if logger.handlers:
        return logger

    # 2. 포맷 설정 (시간 - 레벨 - 메시지)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 3. [Handler 1] 콘솔 출력
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # 4. [Handler 2] 파일 저장 (Runpod의 경우 /workspace에 저장해야 안전)
    log_dir = "/workspace/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, "server.log"))
    file_handler.setLevel(logging.INFO) # 파일엔 INFO 이상만 중요하게 기록
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 싱글톤처럼 사용
logger = setup_logger()