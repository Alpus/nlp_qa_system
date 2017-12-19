import os

SETTINGS_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(SETTINGS_DIR, '../../data')
TEXTS_DIR = os.path.join(DATA_DIR, 'texts')

WORD2VEC_WEIGHTS_FILE = os.path.join(DATA_DIR, os.environ['W2V_WEIGHTS_FILE'])
