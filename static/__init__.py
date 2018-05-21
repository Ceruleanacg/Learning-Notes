import os

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
