import tensorflow as tf
import os
import re

from glob import glob
from static import DATA_DIR

text_paths = glob(os.path.join(DATA_DIR, 'flowers', 'text_c10', '*', '*.txt'))

for text_path in text_paths:
    with open(text_path, 'r') as fp:
        content = fp.read()
        content = re.sub(re.compile('[^a-zA-Z]'), ' ', content)
