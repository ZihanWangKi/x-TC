import itertools
import operator
import os

import numpy as np

retval = os.getcwd()
DATA_FOLDER_PATH = os.path.join(retval, 'datasets')
INFERENCE_PATH = os.path.join(retval, 'inference')
MODEL_PATH = os.path.join(retval, 'models')
os.system(f"mkdir -p {MODEL_PATH}")