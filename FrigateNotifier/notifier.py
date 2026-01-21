import json
import logging
import queue
import threading
import time

from . import CONFIG

class Notifier:
    def __init__(self):
        self._queue = queue.Queue()
        self._mqtt = None
        self._last_notification = {}
        self._time_lock = threading.Lock()
        self._message_thread = threading.Thread(target=self._notify_loop, daemon=True)
        self._message_thread.start()

    def set_client(self, mqtt_client):
        self._mqtt = mqtt_client

    def _notify_loop(self):
        logging.info("Starting notify thread")
        while True:
            try:
                topic, item = self._queue.get()
                cur_time = time.time()

                #  Throttle notifications to one every 5 seconds
                with self._time_lock:
                    if cur_time - self._last_notification.get(topic, 0) < 5:
                        self._queue.task_done()
                        continue

                    self._last_notification[topic] = cur_time
                if not CONFIG.NOTIFY:
                    self._queue.task_done()
                    continue

                if self._mqtt:
                    try:
                        payload = json.dumps(item)
                        # result = self._mqtt.publish(CONFIG.PUB_TOPIC, payload)
                        result = self._mqtt.publish(topic, item)
                        status = result[0]
                        if status == 0:
                            logging.info("Posted MQTT Notification")
                        else:
                            logging.warning(f"Failed posting MQTT notification. Result: {result}")
                    except (TypeError, ValueError) as e:
                        logging.error(f"Error serializing or publishing payload: {e}")
                else:
                    logging.warning("MQTT client not set; cannot publish notification.")

                self._queue.task_done()
            except Exception as e:
                logging.error(f"Unexpected error in notify loop: {e}")

    def __call__(self, obj_type, detection_type, confidence, camera_name = None):
        if camera_name is None:
            camera_name = CONFIG.CAMERA_NAME

        notification = {
            'camera': camera_name,  # Replace with your actual instance identifier
            'label': obj_type,
            'type': detection_type,
            'conf': confidence,
        }

        self._queue.put(('cameramon/object', 'detected'))
        
    def send_custom(self, message, topic):
        self._queue.put((topic, message))

notify = Notifier()
