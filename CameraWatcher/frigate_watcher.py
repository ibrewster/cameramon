import argparse
import importlib.util
import json
import logging
import os
import queue
import requests
import sys
import threading
import time

from datetime import datetime

from paho.mqtt import client as mqtt_client

def load_config(config_file):
    """
    Loads configuration parameters from a specified Python file.

    Args:
      config_file (str): Path to the configuration file.

    Returns:
      module: The imported module containing configuration variables.

    Raises:
      ImportError: If the specified config file cannot be imported.
    """
    spec = importlib.util.spec_from_file_location("config", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_config_path(args):
    """
    Determines the path to the configuration file based on arguments.

    Args:
      args (argparse.Namespace): Parsed command-line arguments.

    Returns:
      str: Path to the configuration file.
    """
    if args.config:
        return args.config
    else:
        script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        config_path = os.path.join(script_dir, "config.py")
        if os.path.exists(config_path):
            return config_path
        else:
            raise FileNotFoundError(
          "No config.py found in script directory and -c flag not provided."
      )

class WaitSet(set):
    """
    A thread-safe set that provides a mechanism to wait for items to be added.

    This class extends the built-in `set` to include functionality that allows
    waiting for an item to be added to the set. It uses a multiprocessing Event
    to signal when an item is added. If the set is empty when `wait()` is called,
    it will block until an item is added.

    Methods:
        add(item):
            Adds an item to the set and sets the event flag.

        remove(item):
            Removes an item from the set and clears the event flag if the set is empty.

        discard(item):
            Removes an item from the set if it exists, and clears the event flag if the set is empty.

        wait() -> bool:
            Blocks until an item is added to the set, returning True if the set is not empty.

        clear():
            Clears the set and the event flag.

    Example:
        >>> my_set = WaitSet()
        >>> my_set.add(1)
        >>> my_set.wait()  # Returns True immediately since 1 is in the set
        >>> my_set.remove(1)
        >>> my_set.wait()  # Blocks until an item is added again
    """
    def __init__(self):
        super().__init__()
        self._event = threading.Event()

    def add(self, item):
        super().add(item)
        self._event.set()

    def remove(self, item):
        super().remove(item)  # Call the parent class's remove method
        self._check_empty()

    def discard(self, item):
        super().discard(item)  # Call the parent class's discard method
        self._check_empty()

    def wait(self, timeout=None):
        """Wait until the set is not empty, or until timeout occurs.

        Args:
            timeout (float): The maximum time to wait in seconds. If None, will block indefinitely.

        Returns:
            bool: True if the set is not empty, False if the timeout occurred.
        """
        # If the set is not empty, return True immediately
        if self:
            return True

        # Block until an item is added
        return self._event.wait(timeout)

    def clear(self):
        super().clear()  # Clear the set
        self._event.clear()  # Clear the event flag

    def _check_empty(self):
        # Clear the event if the set is empty
        if not self:
            self._event.clear()  # Clear the event flag if the set is empty

DELIVERY_SERVICES = frozenset(("usps", "ups", "fedex", "amazon", "dhl"))

class FrigateObject:
    def __init__(self, payload):
        self._moving = False # default to false
        self.id = payload['id']
        self.ids = {payload['id']}
        self.type = payload['label']
        self.conf = payload['score']
        self.box = payload['box']

        self.is_moving = not payload['stationary']

    @property
    def is_moving(self) -> bool:
        return self._moving

    @is_moving.setter
    def is_moving(self, value: bool) -> None:
        if not type(value) == bool:
            raise ValueError("Value for Moving must be a boolean!")

        if value is True and self._moving == False:
            notify(self.type, 'moving', self.conf) # Notify immediately upon object starting motion
            moving_objects.add(self.id)
        elif not value and self._moving:
            moving_objects.discard(self.id)

        self._moving = value

    def update(self, payload):
        if payload['id'] != self.id or payload['label'] != self.type:
            raise ValueError("Cannot update ID or type of an existing object")

        self.is_moving = not payload['stationary']
        self.conf = payload['score']
        self.box = payload['box']

    def __str__(self):
        return f"Object ID: {self.id}, Type: {self.type}, Confidence: {self.conf:.2f}%, Moving: {'Yes' if self._moving else 'No'}"

    def __repr__(self):
        return f"FrigateObject(payload={{'id': {self.id}, 'label': '{self.type}', 'score': {self.conf}, 'stationary': {not self._moving}, 'box': {self.box}}})"

class Notifier:
    def __init__(self):
        self._queue = queue.Queue()
        self._mqtt = None
        self._last_notification = {}  # Keep as dict for per-topic tracking
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

                # Throttle notifications per-topic to one every 5 seconds
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
                        payload = item if isinstance(item, str) else json.dumps(item)
                        result = self._mqtt.publish(topic, payload)
                        status = result[0]
                        if status == 0:
                            logging.info(f"Posted MQTT Notification to {topic}")
                        else:
                            logging.warning(f"Failed posting MQTT notification to {topic}. Result: {result}")
                    except (TypeError, ValueError) as e:
                        logging.error(f"Error serializing or publishing payload: {e}")
                else:
                    logging.warning("MQTT client not set; cannot publish notification.")

                self._queue.task_done()
            except Exception as e:
                logging.error(f"Unexpected error in notify loop: {e}")

    def __call__(self, obj_type, detection_type, confidence, camera_name=None):
        if camera_name is None:
            camera_name = CONFIG.CAMERA_NAME
            
        message = {
            'camera': camera_name,
            'label': obj_type,
            'type': detection_type,
            'conf': confidence,
        }

        self._queue.put((CONFIG.PUB_TOPIC, message))
        
    def send_custom(self, message, topic=None):
        if topic is None:
            topic = CONFIG.PUB_TOPIC

        self._queue.put((topic, message))

def MotionMonitor():
    """
    Monitors the movement status of objects and sends notifications when any object is moving.

    This function runs in a separate thread and follows these key behaviors:
    - Waits for any objects to be marked as moving. If no objects are moving, it will not perform any checks.
    - Once any object is detected as moving, it will notify every 2 seconds about the moving objects.
    - If an object is removed from the known objects, it is also removed from the set of moving objects for safety.
    - Ensures that notifications are only sent for valid objects, handling any discrepancies in object status gracefully.

    The function will run indefinitely until explicitly terminated.
    """
    # Run a loop to send notifications periodically while anything is moving.
    logging.info("Starting motion monitor thread")
    while True:
        moving_objects.wait() # Do nothing unless something is moving.

        types = []
        confidences = []
        orphaned_ids = []

        for _id in moving_objects:
            item = known_objects.get(_id)
            if item is not None:
                types.append(item.type)
                confidences.append(item.conf)
            else:
                # if item does not exist in known_objects, it should not be in moving_objects
                # It should have been removed elsewhere, but go ahead and mark it for removal here
                logging.warning(f"Orphaned item id {_id} found in moving_objects. Discarding it.")
                orphaned_ids.append(_id)

        if types:
            logging.info("MOVING: One or more objects is moving. Notifying.")
            notify(types, 'motion', confidences)

        for _id in orphaned_ids:
            moving_objects.discard(item)

        time.sleep(2) # Throttle this loop to run not more often than once every two seconds.

    logging.info("Motion monitoring terminated")

def on_disconnect(client, userdata, disconnect_flags, reason, properties):
    # Determine if the disconnect reason might be resolved with a retry
    retryable_flags = [0, 2, 3, 6, 7, 16]
    if disconnect_flags in retryable_flags:
        logging.warning(f"Disconnected from MQTT broker: {reason}")
        # Retry a fixed number of times with a variable delay
        max_retries = 3
        retry_delay = 1
        for i in range(max_retries):
            logging.info(f"Retrying MQTT connection (attempt {i+1}/{max_retries})")
            client.reconnect()
            if client.is_connected():
                logging.info("Reconnect attempt succeeded.")
                break
            else:
                rc = client.reconnect_result()
                error_string = mqtt_client.error_string(rc)
                logging.warning(f"Retry attempt {i+1} failed. Reason: {error_string}")

            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        else:
            logging.error("Failed to reconnect to MQTT broker after multiple attempts")
            raise SystemExit("Unable to connect to MQTT broker")
    else:
        logging.critical(f"Fatal error: Disconnected from MQTT broker: {reason}. Not retrying.")
        raise SystemExit("Unable to connect to MQTT broker")


def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        logging.info("Connected to MQTT Broker.")
    else:
        logging.warning("Failed to connet, return code: %s", mqtt_client.error_string(rc))

def connect_mqtt():
    broker = 'conductor.brewstersoft.net'
    current_time = str(int(time.time()))[:-3]
    client_id = f'frigate-mqtt-monitor-{current_time}'
    username = 'hamqtt'
    password = 'Sh@nima821'

    client = mqtt_client.Client(
        client_id=client_id,
        callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2
    )
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.connect(broker)
    return client

# Function to get a snapshot from Frigate
def get_snapshot(camera_name=None, annotated=False):
    if camera_name is None:
        camera_name = CONFIG.CAMERA_NAME
    # Determine the appropriate URL for the snapshot
    url = f"http://watchman.brewstersoft.net:5005/api/{camera_name}/latest.jpg"
    params = {
        'bbox': '1' if annotated else '0',  # Bounding box for annotations
        'h': '1080'  # Image height
    }

    # Request the snapshot
    try:
        # Make the request with the query parameters
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        logging.info(f"Snapshot retrieved from {camera_name} (annotated: {annotated})")
        return response.content  # Return image data
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to get snapshot from {camera_name}: {e}")
        return None

def gen_json():
    detect_info = {
        'labels': [],
        'boxes': [],
        'frame_id': 'snapshot',
        'confidences': [],
        'image_dimensions': {
            'original': [1920, 1080],
            'resized': [1920, 1080],
        },
    }

    for obj in known_objects.values():
        detect_info['labels'].append(obj.type)
        detect_info['confidences'].append(obj.conf)
        detect_info['boxes'].append(obj.box)

    return detect_info

def save_annotations(payload):
    item_id = payload['after']['id']

    json_dir = os.path.join(CONFIG.FILE_DIR, 'frigate')
    os.makedirs(json_dir, exist_ok=True)
    json_file = os.path.join(json_dir, f"{item_id}.json")
    with open(json_file, 'w') as file:
        json.dump(payload, file)

    now = datetime.now()
    img_dir = os.path.join(CONFIG.FILE_DIR, CONFIG.CAMERA_NAME, now.strftime('%Y-%m-%d'))
    os.makedirs(img_dir, exist_ok=True)

    time_str = now.strftime('%H-%M-%S')
    annotated_filename = os.path.join(img_dir, f"{time_str}_objdetect.jpg")
    clean_filename = os.path.join(img_dir, f"{time_str}_clean.jpg")
    json_filename = os.path.join(img_dir, f"{time_str}_objects.json")

    annotated_snapshot = get_snapshot(annotated=True)
    if annotated_snapshot:
        with open(annotated_filename, 'wb') as annotated_file:
            annotated_file.write(annotated_snapshot)
        logging.info(f"Annotated snapshot saved to: {annotated_filename}")

        with open(json_filename, 'w') as jf:
            json.dump(gen_json(), jf)

    # Get clean snapshot
    clean_snapshot = get_snapshot(annotated=False)
    if clean_snapshot:
        with open(clean_filename, 'wb') as clean_file:
            clean_file.write(clean_snapshot)
        logging.info(f"Clean snapshot saved to: {clean_filename}")

def boxes_match(box1, box2, iou_thresh=0.5, max_center_dist=20):
    from math import sqrt

    # IoU
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        iou_val = 0.0
    else:
        inter_area = (x2 - x1) * (y2 - y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        iou_val = inter_area / (area1 + area2 - inter_area)

    # Center distance
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    center_dist = sqrt((cx2-cx1)**2 + (cy2-cy1)**2)

    return iou_val >= iou_thresh or center_dist <= max_center_dist

def match_existing(after):
    for existing in known_objects.values():
        if existing.type != after['label']:
            continue
        if boxes_match(existing.box, after['box']):  # IoU or center-distance threshold
            return existing
    return None

def on_message(client, userdata, msg):
    json_payload = msg.payload.decode()
    payload = json.loads(json_payload)

    after = payload['after']
    item_id = after['id']
    item_type = after['label']

    # Make sure this isn't a false positive. Ignore it if so.
    if after['false_positive'] == True:
        return

    # See if we need to remove this object (end time set)
    if after['end_time'] is not None:
        try:
            known_objects[item_id].ids.remove(item_id)
            del known_objects[item_id]
            moving_objects.discard(item_id)
            logging.info(f"Removed item id: {item_id} of type: {item_type}")
        except KeyError:
            pass
        return

    # see if this is a new object
    if item_id not in known_objects:        
        if not after['current_zones']:
            # ignore the object if not in any zones
            logging.debug(f"Ignoring {item_type} as it is not in the zones")
            return

        # See if we think this is the same object, just with a different ID
        existing = match_existing(after) # object or None
        if existing:
            existing.ids.add(item_id)
            known_objects[item_id] = existing
            logging.info(f"Re-acquired {item_type} with new id {item_id}")
            return

        # add the object to our list
        logging.info(f"NEW {item_type}, {after['score'] * 100:.2f}%: Adding {item_type} with id {item_id} to the tracked list")
        obj = FrigateObject(after)
        known_objects[item_id] = obj

        # Save the payload
        save_annotations(payload)

        # Check for fancy stuff
        delivery_vehicle = False
        if item_type == 'car':
            sub_label = after.get("sub_label")
            if sub_label and isinstance(sub_label, list):
                delivery_vehicle = sub_label[0] in DELIVERY_SERVICES

        if item_type == 'package' or delivery_vehicle:
            logging.info("!!!PACKAGE DELIVERY!!!")
            notify.send_custom('detected', topic='cameramon/delivery')

        if obj.is_moving:
            notify(obj.type, 'new', obj.conf)
        else:
            # If it's a "new" object, but is stationary, don't notify about it, just start tracking.
            logging.info("Not notifying due to stationary object.")
    else:
        obj = known_objects.get(item_id, FrigateObject(after))
        obj.update(after)



############## GLOBAL OBJECTS ################
known_objects = {
    # Key: object id. Value: object object
}

notify = Notifier()
moving_objects = WaitSet()
################################################

if __name__ == "__main__":
    # Import config file
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument(
        "-c", "--config", help="Path to the configuration file (optional)."
    )
    args = parser.parse_args()

    try:
        config_path = get_config_path(args)
        CONFIG = load_config(config_path)
        # Access your configuration values using config object
        # For example, config['DEFAULT']['my_param']
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # configure logging
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(CONFIG.LOG_LEVEL)  # Ensure the handler level is set correctly
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(CONFIG.LOG_LEVEL)  # Set the root logger level
    root_logger.handlers = [stream_handler]  # Replace existing handlers with the new one

    logging.info("Starting monitoring")
    os.makedirs(CONFIG.FILE_DIR, exist_ok=True)

    # Set up the motion notifier
    motion_thread = threading.Thread(target=MotionMonitor, daemon=True)
    motion_thread.start()

    client = connect_mqtt()
    notify.set_client(client)
    client.subscribe(CONFIG.SUB_TOPIC)
    client.loop_forever()