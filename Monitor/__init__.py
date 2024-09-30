import configparser
import logging
import os

from logging import handlers

LOG_LEVEL = logging.INFO
logger = None

def init_logging():
    global logger
    FORMAT = "%(asctime)-15s %(levelname)s: %(message)s"
    logger = logging.getLogger("cameramon")
    logger.setLevel(LOG_LEVEL)

    # File logging
    handler = handlers.RotatingFileHandler(
        '/var/log/cameramon/cameramon.log',
        maxBytes=1024000, backupCount=5)
    fmt = logging.Formatter(FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(fmt)
    handler.setLevel(LOG_LEVEL)
    logger.addHandler(handler)

    logging.basicConfig(format=FORMAT, level=LOG_LEVEL, datefmt='%Y-%m-%d %H:%M:%S')
    
config = configparser.ConfigParser()
MODULE_PATH = os.path.dirname(__file__)
config.read(os.path.join(MODULE_PATH, 'cameramon.ini'))
    
init_logging()
print("My PID is:", os.getpid())

from .CameraMonitor import main as run