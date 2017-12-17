import logging
import sys

__all__ = ['logger']


logger = logging.getLogger('QA')
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

# add the handlers to the logger
logger.addHandler(handler)
