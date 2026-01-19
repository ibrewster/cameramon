import logging
import threading
import time

from .frigate import moving_objects, known_objects
from .notifier import notify

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

def run():
    motion_thread = threading.Thread(target=MotionMonitor, daemon=True)
    motion_thread.start()
