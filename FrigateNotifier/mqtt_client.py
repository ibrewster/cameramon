import json
import logging
import os
import time

from datetime import datetime

from paho.mqtt import client as mqtt_client

from . import frigate, CONFIG
from .notifier import notify
from .utils import save_annotations

DELIVERY_SERVICES = frozenset(("usps", "ups", "fedex", "amazon", "dhl"))

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


def on_message(client, userdata, msg):
    json_payload = msg.payload.decode()
    payload = json.loads(json_payload)

    after = payload['after']
    item_id = after['id']
    item_type = after['label']
    sub_label = after.get("sub_label")    

    # Make sure this isn't a false positive. Ignore it if so.
    if after['false_positive'] == True:
        return

    # See if we need to remove this object (end time set)
    if after['end_time'] is not None:
        try:
            del frigate.known_objects[item_id]
            frigate.moving_objects.discard(item_id)
            logging.info(f"Removed item id: {item_id} of type: {item_type}")
        except KeyError:
            pass
        return

    # see if this is a new object
    if item_id not in frigate.known_objects:
        if not after['current_zones']:
            # ignore the object if not in any zones
            logging.debug(f"Ignoring {item_type} as it is not in the zones")
            return

        # add the object to our list
        logging.info(f"NEW {item_type}, {after['score'] * 100:.2f}%: Adding {item_type} with id {item_id} to the tracked list")
        obj = frigate.FrigateObject(after)
        frigate.known_objects[item_id] = obj

        # Check for fancy stuff
        delivery_vehicle = False
        if sub_label:
            if isinstance(sub_label, list):
                delivery_vehicle = sub_label[0] in DELIVERY_SERVICES
            else:
                logging.warning(f"Sub label {sub_label} is not a list!")                

        if item_type == 'package' or delivery_vehicle:
            logging.info("!!!PACKAGE DELIVERY!!!")
            notify.send_custom('detected', 'cameramon/delivery')

        if obj.is_moving:
            # Save the payload
            save_annotations(payload)
            notify(obj.type, 'new', obj.conf)
        else:
            # If it's a "new" object, but is stationary, don't notify about it, just start tracking.
            logging.info("Not notifying due to stationary object.")
    else:
        obj = frigate.known_objects.get(item_id, frigate.FrigateObject(after))
        obj.update(after)


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

def loop_forever():
    client = connect_mqtt()
    notify.set_client(client)
    client.subscribe(CONFIG.SUB_TOPIC)
    client.loop_forever()
