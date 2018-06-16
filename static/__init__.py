import os

SUMR_DIR = os.path.join(os.path.dirname(__file__), 'summaries')
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
