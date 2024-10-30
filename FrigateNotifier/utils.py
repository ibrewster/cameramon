import json
import logging
import os
import threading

from datetime import datetime

from . import CONFIG

def setup_logging(log_level):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)  # Ensure the handler level is set correctly
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)  # Set the root logger level
    root_logger.handlers = [stream_handler]  # Replace existing handlers with the new one


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




def save_annotations(payload):
    from .frigate import get_snapshot

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

def gen_json():
    from .frigate import known_objects
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
