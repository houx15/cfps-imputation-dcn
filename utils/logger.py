import logging
import logging.handlers
import os
import time

from dotenv import load_dotenv


def get_logger() -> logging.Logger:
    """获取logger

    Returns:
        logging.Logger: logger
    """
    load_dotenv()
    log_root = os.getenv("LOG_ROOT")
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = os.path.join(log_root, f"dcn-{curr_time}.log")
    logger = logging.getLogger("DCN")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # 按照大小切割文件
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=1024 * 1024 * 5, backupCount=1024 * 1024
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


LOGGER = get_logger()
