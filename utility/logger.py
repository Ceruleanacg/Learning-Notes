import logging
import os

from datetime import datetime
from static import LOGS_DIR
from time import time

DATETIME_NOW = datetime.now().strftime("%Y%m%d%H%M%S")


def get_logger(model_name, mode, filename):
    # Make path.
    dir_path = os.path.join(LOGS_DIR, model_name, mode)
    log_path = os.path.join(dir_path, '{}-{}.log'.format(DATETIME_NOW, filename))
    # Check path.
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # Get logger.
    logger_name = model_name + '-' + filename
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Get logger stream handler.
    # log_sh = logging.StreamHandler(sys.stdout)
    log_sh = logging.StreamHandler()
    # log_sh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
    log_sh.setLevel(logging.WARNING)
    # Get logger file handler.
    log_fh = logging.FileHandler(log_path)
    log_fh.setLevel(logging.DEBUG)
    log_fh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
    # Add handler.
    logger.addHandler(log_sh)
    logger.addHandler(log_fh)
    return logger


class TimeInspector(object):

    time_marks = []

    @classmethod
    def set_time_mark(cls):
        _time = time()
        cls.time_marks.append(_time)
        return _time

    @classmethod
    def pop_time_mark(cls):
        cls.time_marks.pop()

    @classmethod
    def log_cost_time(cls, info):
        cost_time = time() - cls.time_marks.pop()
        logging.warning('Time cost: {0:.2f} | {1}'.format(cost_time, info))
