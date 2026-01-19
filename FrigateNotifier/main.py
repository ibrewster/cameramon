from .utils import setup_logging
from . import CONFIG, mqtt_client, motion_monitor

import logging
import os

def run():
    # configure logging
    setup_logging(CONFIG.LOG_LEVEL)

    logging.info("Starting monitoring")

    # Make sure the directory to contain output files exists
    os.makedirs(CONFIG.FILE_DIR, exist_ok=True)

    # Set up the motion notifier
    motion_monitor.run()

    # And run the MQTT Client monitoring loop
    mqtt_client.loop_forever()


if __name__ == "__main__":
    run()
