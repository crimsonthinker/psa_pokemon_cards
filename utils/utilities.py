import logging
import os

def get_logger(logger_namescope):
    log_format = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) -35s %(lineno) -5d: %(message)s')
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(logger_namescope)

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)