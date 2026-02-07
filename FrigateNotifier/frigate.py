import logging
import time

import requests

from . import CONFIG
from .notifier import notify

from .utils import WaitSet

known_objects = {
    # Key: object id. Value: object object
}

moving_objects = WaitSet()

class FrigateObject:
    def __init__(self, payload):
        self._moving = False # default to false
        self.id = payload['id']
        self.type = payload['label']
        self.conf = payload['score']
        self.box = payload['box']
        self.delivery = (payload.get("sub_label") or [None])[0]
        self.created = time.time()

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
        if not self.delivery and (time.time() - self.created) < 5:
            self.delivery = (payload.get("sub_label") or [None])[0]

    def __str__(self):
        return f"Object ID: {self.id}, Type: {self.type}, Confidence: {self.conf:.2f}%, Moving: {'Yes' if self._moving else 'No'}"

    def __repr__(self):
        return f"FrigateObject(payload={{'id': {self.id}, 'label': '{self.type}', 'score': {self.conf}, 'stationary': {not self._moving}, 'box': {self.box}}})"


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
