import configparser
import os

import flask

MODULE_PATH = os.path.dirname(__file__)
CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(MODULE_PATH, 'cameramon.ini'))
app = flask.Flask(__name__)

from . import main
