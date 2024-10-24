import json
import logging
import os
import requests
import threading
import time

from datetime import datetime

from paho.mqtt import client as mqtt_client

LOG_LEVEL = logging.INFO
FILE_DIR = '/data/cameramon'
# FILE_DIR = '/Users/israel/Desktop/cameramon'
CAMERA_NAME = 'drivecam'

class FrigateObject:
    def __init__(self, payload):
        self._moving = False
        self.id = payload['id']
        self.type = payload['label']
        self.conf = payload['score']
        self.box = payload['box']
        
    @property
    def is_moving(self) -> bool:
        return self._moving
    
    @is_moving.setter
    def is_moving(self, value: bool) -> None:
        if not type(value) == bool:
            raise ValueError("Value for Moving must be a boolean!")
        if value is True and self._moving == False:
            notify() # Notify immediately upon object starting motion
            
        self._moving = value
 
class Notifier:
    def __init__(self):
        self._mqtt = None
        self._last_notification = 0
        self._time_lock = threading.Lock()
        self._notify = threading.Event()
        self._message_thread = threading.Thread(target=self._notify_loop, daemon=True)
        self._message_thread.start()
        
    def set_client(self, mqtt_client):
        self._mqtt = mqtt_client
        
    def _notify_loop(self):
        logging.info("Starting notify thread")
        while True:
            self._notify.wait()
            self._notify.clear()
            cur_time = time.time()
            
            #  Throttle notifications to one every 5 seconds            
            with self._time_lock:
                if cur_time - self._last_notification < 5:
                    continue
            
                self._last_notification = cur_time
        
            if self._mqtt:
                result = self._mqtt.publish('cameramon/object', 'detected')
                status = result[0]
                if status == 0:
                    logging.info("Posted MQTT Notification")
                else:
                    logging.warning(f"Failed posting MQTT notification. Result: {result}")
                    
            # try:
                # result = requests.get('http://10.27.81.71:5000/camview')
                # result.raise_for_status()
                # logging.info("URL notified")
            # except Exception as e:
                # logging.warning(f"Unable to call URL: {e}")  
        
    def __call__(self):
        self._notify.set()
    
known_objects = {
    # Key: object id. Value: object object
}

notify = Notifier()

def MotionMonitor():
    # Run a loop to send notifications periodically while anything is moving.
    logging.info("Starting motion monitor thread")
    while True:
        time.sleep(2)
        for item in known_objects.values():
            if item.moving:
                logging.info("MOVING: One or more objects is moving. Notifying.")
                notify()
                continue
    logging.info("Motion monitoring terminated")

FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60
    
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
    client_id = 'frigate-mqtt-monitor'
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
def get_snapshot(camera_name=CAMERA_NAME, annotated=False):
    # Determine the appropriate URL for the snapshot
    url = f"http://watchman.brewstersoft.net:5005/api/{CAMERA_NAME}/latest.jpg"
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
    
    json_dir = os.path.join(FILE_DIR, 'frigate')
    os.makedirs(json_dir, exist_ok=True)
    json_file = os.path.join(json_dir, f"{item_id}.json")
    with open(json_file, 'w') as file:
        json.dump(payload, file)
        
    now = datetime.now()
    img_dir = os.path.join(FILE_DIR, CAMERA_NAME, now.strftime('%Y-%m-%d'))
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
            del known_objects[item_id]
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
    
            
        # add the object to our list
        logging.info(f"NEW {item_type}, {after['score'] * 100:.2f}%: Adding {item_type} with id {item_id} to the tracked list")
        obj = FrigateObject(after)
        known_objects[item_id] = obj
        
        # Save the payload
        save_annotations(payload)        
        
        if 'box' in after['attributes']:
            logging.info("!!!PACKAGE DELIVERY!!!")
            
        notify()

    try:
        # if existing object, update motion. Also updates "new" objects, so DRY
        known_objects[item_id].moving = not after['stationary']
    except KeyError:
        logging.warning(f"Unable to update object of type {item_type} as it is not in the known_objects dict!")
    

if __name__ == "__main__":
    # configure logging
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)  # Ensure the handler level is set correctly
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)  # Set the root logger level
    root_logger.handlers = [stream_handler]  # Replace existing handlers with the new one

    logging.info("Starting monitoring")
    os.makedirs(FILE_DIR, exist_ok=True)
    
    # Set up the motion notifier
    motion_thread = threading.Thread(target=MotionMonitor, daemon=True)
    motion_thread.start()
    
    topic = 'frigate/events'
    client = connect_mqtt()
    notify.set_client(client)
    client.subscribe(topic)
    client.loop_forever()