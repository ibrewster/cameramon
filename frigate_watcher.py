import json
import logging
import threading
import time

import requests

from paho.mqtt import client as mqtt_client

class FrigateObject:
    def __init__(self, item_id, item_type):
        self._moving = False
        self.id = item_id
        self.type = item_type
        
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
        print("Running notify thread")
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
                    print("Posted MQTT Notification")
                else:
                    print(f"Failed posting MQTT notification. Result: {result}")
                    
            # try:
                # result = requests.get('http://10.27.81.71:5000/camview')
                # result.raise_for_status()
                # print("URL notified")
            # except Exception as e:
                # print(f"Unable to call URL: {e}")  
        
    def __call__(self):
        self._notify.set()
    
known_objects = {
    # Key: object id. Value: object object
}

notify = Notifier()

def MotionMonitor():
    # Run a loop to send notifications periodically while anything is moving.
    print("Starting motion monitor thread")
    while True:
        time.sleep(2)
        for item in known_objects.values():
            if item.moving:
                print("MOVING: One or more objects is moving. Notifying.")
                notify()
                continue
    print("Motion monitoring terminated")

FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60
    
def on_disconnect(client, userdata, rc):
    logging.info("Disconnected with result code: %s", rc)
    reconnect_count, reconnect_delay = 0, FIRST_RECONNECT_DELAY
    while reconnect_count < MAX_RECONNECT_COUNT:
        logging.info("Reconnecting in %d seconds...", reconnect_delay)
        time.sleep(reconnect_delay)

        try:
            client.reconnect()
            logging.info("Reconnected successfully!")
            return
        except Exception as err:
            logging.error("%s. Reconnect failed. Retrying...", err)

        reconnect_delay *= RECONNECT_RATE
        reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
        reconnect_count += 1
    logging.info("Reconnect failed after %s attempts. Exiting...", reconnect_count)
    
def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print("Connected to MQTT Broker.")
    else:
        print("Failed to connet, return code:", rc)

def connect_mqtt():
    broker = 'conductor.brewstersoft.net'
    client_id = 'frigate-mqtt-monitor'
    username = 'hamqtt'
    password = 'Sh@nima821'
    
    client = mqtt_client.Client(client_id=client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.connect(broker)
    return client

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    payload = json.loads(payload)
    after = payload['after']
    item_id = after['id']
    item_type = after['label']
    
    # See if we need to remove this object (end time set)
    if after['end_time'] is not None:
        try:
            del known_objects[item_id]
            print(f"Removed item id: {item_id} of type: {item_type}")            
        except KeyError:
            pass
        return

    # see if this is a new object
    if item_id not in known_objects:
        if not after['current_zones']:
            # ignore the object if not in any zones
            # print(f"Ignoring {item_type} as it is not in the zones")
            return
        
        # add the object to our list
        print(f"NEW {item_type}: Adding {item_type} to the tracked list")
        obj = FrigateObject(item_id, item_type)
        known_objects[item_id] = obj
        
        if 'box' in after['attributes']:
            print("!!!PACKAGE DELIVERY!!!")
            
        notify()

    try:
        # if existing object, update motion. Also updates "new" objects, so DRY
        known_objects[item_id].moving = not after['stationary']
    except KeyError:
        print(f"Unable to update object of type {item_type} as it is not in the known_objects dict!")
    

if __name__ == "__main__":
    # Set up the motion notifier
    motion_thread = threading.Thread(target=MotionMonitor, daemon=True)
    motion_thread.start()
    
    topic = 'frigate/events'
    client = connect_mqtt()
    notify.set_client(client)
    client.subscribe(topic)
    client.loop_forever()