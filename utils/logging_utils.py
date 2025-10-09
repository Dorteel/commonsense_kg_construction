import logging

def setup_logger(name="project"):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s"
    )
    return logging.getLogger(name)