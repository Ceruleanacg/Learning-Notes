import logging
import os

from datetime import datetime
from static import LOGS_DIR

DATETIME_NOW = datetime.now().strftime("%Y%m%d%H%M%S")


def generate_model_logger(model_name):

    log_path = '{}-{}-{}'.format(model_name, DATETIME_NOW, '.log')

    logger = logging.getLogger('model_logger')
    logger.setLevel(logging.DEBUG)

    log_sh = logging.StreamHandler()
    log_sh.setLevel(logging.WARNING)

    log_fh = logging.FileHandler(os.path.join(LOGS_DIR, log_path))
    log_fh.setLevel(logging.DEBUG)
    log_fh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))

    logger.addHandler(log_sh)
    logger.addHandler(log_fh)

    return logger
