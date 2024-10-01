import logging
import os
import re
import socket
import time

from collections import defaultdict
from datetime import datetime, timedelta

import cv2
import numpy
import requests
import ujson as json

import paho.mqtt.publish as publish_mqtt
from PIL import Image
from shapely.geometry import box

coral_tpu = False

try:
    from pycoral.adapters.common import input_size, input_tensor
    from pycoral.adapters.detect import get_objects
    from pycoral.utils.dataset import read_label_file
    from pycoral.utils.edgetpu import make_interpreter
    coral_tpu = True
except ImportError:
    from .tf_funcs import (
        read_label_file,
        input_size,
        input_tensor,
        get_objects,
        make_interpreter
    )

from .video import VideoCapture, Tracker
from .zoneminder import load_zones
from . import config

interpreter = None
inference_size = None
labels = None
zones = None
car_zones = None
logger = None
trackers = defaultdict(list)
detect_pattern = re.compile('(person|car|motorbike|bus|truck|boat|skateboard|horse|dog|cat)')
interesting_objects = ("person", "bicycle", "car", "motorbike", "bus", "truck", "boat", "skateboard", "horse", "dog", "cat")
logger = logging.getLogger("cameramon")

def init():
    global interpreter
    global labels
    global inference_size
    global zones
    global car_zones

    logger.info("Setting up model")
    interpreter = make_interpreter(config['model']['object_weights'])

    interpreter.allocate_tensors()
    labels = read_label_file(config['model']['object_labels'])
    inference_size = input_size(interpreter)

    logger.info("Loading Zones")
    zones, car_zones = load_zones(3)



def run_inference(image):
    logger.debug(f"Detecting objects in image")

    h, w, channel = image.shape
    input_tensor(interpreter)[:h, :w] = image
    interpreter.invoke()

    objs = get_objects(interpreter, config.getfloat('model', 'min_conf'))
    return objs


def update_trackers(frame):
    """
    Update existing trackers with the current frame.

    Args:
        frame: The current frame from the video stream.

    Returns:
        bool: True if any trackers were successfully updated, False otherwise.
    """
    update_time = time.time()
    for class_id, tracker_list in list(trackers.items()):
        for i in range(len(tracker_list) - 1, -1, -1):  # Iterate in reverse to allow safe removal
            tracker = tracker_list[i]
            if update_time - tracker.last_seen > 20:
                # object hasn't been seen in the past 20 seconds. Remove tracker
                # Detection runs at something like 10 fps, so if not touched
                # in a full second, that's 10 detections in a row where it
                # hasn't matched anything.
                logger.warning("Stale tracker detected. Removing!")
                trackers[class_id].pop(i)
                continue

            success, updated_bbox = tracker.update(frame)

            if not success:
                logger.warning("Tracker thinks object gone. Removing from list")
                # Tracker lost, remove it from the list
                trackers[class_id].pop(i)


def calculate_iou(shapeA, shapeB):
    """
    Calculate the Intersection over Union (IoU) between two Shapely geometries.

    The IoU is a measure of the overlap between two shapes, calculated as the area of
    their intersection divided by the area of their union.

    Parameters:
    shapeA (shapely.geometry.Polygon or shapely.geometry.box): The first shape.
    shapeB (shapely.geometry.Polygon or shapely.geometry.box): The second shape.

    Returns:
    float: The IoU value between shapeA and shapeB. A value of 1.0 means perfect overlap,
           while a value of 0.0 means no overlap.
    """
    # Calculate the intersection area
    intersection_area = shapeA.intersection(shapeB).area

    # Calculate the union area
    union_area = shapeA.union(shapeB).area

    # Return the IoU (Intersection over Union)
    return intersection_area / union_area if union_area > 0 else 0

def detect_image(image, img_ratio, img_area):
    update_trackers(image)

    logger.debug("Processing image for event")
    signal = False
    good_objects = []

    detected_objs = run_inference(image)
    logger.debug(f"{detected_objs}")

    if not detected_objs:
        return (signal, good_objects)

    logger.debug("Object detection complete. Processing results")

    objects = []
    confs = []
    bboxes = []
    for item in detected_objs:
        obj = labels[item.id]
        objects.append(obj)

        bbox = numpy.asarray(item.bbox) * img_ratio
        bboxes.append(bbox)

        conf = item.score
        confs.append(conf)

        if obj not in interesting_objects:
            logger.debug(f"Ignoring {obj} {bbox} as it does not match our detect patern")
            continue  # Not an object of interest to us

        bbox_poly = box(*bbox)
        detected_bbox = box(*item.bbox)

        poly_area = bbox_poly.area
        if poly_area / img_area > config.getfloat('match', 'max_det_size'):
            logger.debug(f"Ignoring {obj} {bbox_poly.bounds} as it is too large")
            continue

        # We look at a more restrictive set of zones if the object is a vehicle
        # Hopefully will avoid false alarms of a car in the yard...
        check_zones = car_zones if obj in ['car', 'truck'] else zones
        zone_name = "car zones" if check_zones is car_zones else "zones"
        logger.debug(f"Checking object of type {obj} against zones {zone_name}")

        # If the object is not inside our zones of interest, ignore it.
        if not check_zones.intersects(bbox_poly):
            logger.debug(f"Ignoring {obj} {bbox_poly.bounds} as it is outside our zones")
            continue

        # If the object is a car or truck, and it is mostly in the yard, ignore it even if
        # it pokes into the driveway
        full_width = image.shape[1] * img_ratio
        vehicles = ('car', 'truck', 'bus')
        if obj in vehicles and bbox[2] > (full_width - 100):
            logger.debug(f"Got car in yard (bbox: {bbox}, image width: {full_width}). Ignoring.")
            continue

        # If we get here, list as a good object
        good_objects.append((obj, bbox, conf))

        if obj in vehicles:
            # Compare to *any* vehicles we have seen, as it often gets confused
            comp_objects = [shape
                            for vehicle in vehicles
                            for shape in trackers.get(vehicle, [])]
        else:
            comp_objects = trackers.get(obj, []).copy()

        if not comp_objects:
            # No trackers for this type of object
            max_iou = -1
        elif len(detected_objs) == 1 and len(comp_objects) == 1:
            # If there is only a single object detected, and only a single
            # tracker for this type of object, assume they are the same.
            max_iou = 1
            tracker = comp_objects[0]
        else:
            logging.debug(f"Comp objects is: {comp_objects}, Type: {type(comp_objects)}")
            iou_results = {}
            for idx, tracker in enumerate(comp_objects):
                updated_bbox = tracker.bbox
                # compare the detected object box to the current box from the tracker
                # to see if it is the same object.
                iou = calculate_iou(detected_bbox, updated_bbox)
                iou_results[idx] = iou
                logger.debug(f"IoU for object {obj}, {conf:.2f}: {iou:.2f}")

            best_tracker_idx = max(iou_results, key=iou_results.get)
            tracker = comp_objects[best_tracker_idx]
            max_iou = iou_results[best_tracker_idx]


        if max_iou > config.getfloat('match', 'min_iou'):
            # if we have any overlap with an existing tracker, assume it is the same object.
            # In the event of multiple matches, the one with the most overlap will be used.
            # If a new object of the same type suddenly appears with an IoU of at least 0.19,
            # we may have a problem. Time will tell.
            tracker.touch()
            if tracker.is_moving:  # Define your velocity threshold
                logger.warning(f"Object {obj} has moved.")
                signal = True  # The object has moved
        else:
            # IoU indicates no match, therfore new object.
            logger.warning(f"Matching {obj} as it appears to be new. Max iou: {max_iou}")
            signal = True
            new_tracker = Tracker()
            new_tracker.init(image, detected_bbox)
            trackers[obj].append(new_tracker)

    return (signal, good_objects)


def process_image(pil_image):
    if pil_image is None or not any(pil_image.size):
        raise TypeError("Image was not an image")

    t1 = time.time()
    resized_pil,target_ratio,img_area = resize_pil_image(pil_image)
    logger.debug(f"Resized image to {inference_size} in {time.time() - t1}")

    opencv_image = numpy.asarray(resized_pil)

    found_match, all_objects = detect_image(opencv_image, target_ratio, img_area)

    return (bool(found_match), all_objects)


def resize_pil_image(pil_image):
    target_ratio = max(pil_image.size[0] / inference_size[0],
                       pil_image.size[1] / inference_size[1])

    img_area = pil_image.size[0] * pil_image.size[1]
    logger.debug("Resizing image")

    new_size = (numpy.asarray(pil_image.size) // target_ratio).astype(int)
    resized_pil = pil_image.resize(new_size.tolist(), reducing_gap=2.0)
    return (resized_pil, target_ratio, img_area)


def save_image(objects, image):
    # convert image to an openCV image for editing, if needed
    if not isinstance(image, numpy.ndarray):
        # No need to convert color channels. Since PIL was only used
        # for resizing, we just left them in the cv2 order.
        image = numpy.asarray(image).copy()

    eventid = 1
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = .75
    font_thickness = 1
    frame_color = (46, 174, 85)
    padding = 3
    img_date = datetime.now().strftime('%Y-%m-%d')
    img_time = datetime.now().strftime('%H-%M-%S')

    file_path = os.path.join(config['action']['image_directory'], 'Driveway',
                             img_date)

    os.makedirs(file_path, exist_ok = True)

    image_path = os.path.join(file_path, f'{img_time}_objdetect.jpg')
    json_path = os.path.join(file_path, f'{img_time}_objects.json')

    logger.info(f"Saving image to: {image_path}")

    image_link = f'https://watchman.brewstersoft.net/zm/index.php?view=image&eid={eventid}&fid=objdetect'
    detect_info = {
        'labels': [],
        'boxes': [],
        'frame_id': 'snapshot',
        'confidences': [],
        'image_dimensions': {
            'original': image.shape[:2],
            'resized': image.shape[:2],
        },
    }
    note = f"<a href={image_link}> Detected:"
    for obj, bbox, conf in objects:
        detect_info['labels'].append(obj)
        detect_info['confidences'].append(float(conf))
        logger.debug(f"Drawing images for {obj} ({conf}) at {bbox}")
        bbox = bbox.round().astype(int)
        detect_info['boxes'].append(bbox.tolist())
        top_left = bbox[:2]
        bottom_right = bbox[2:]
        label = f"{obj} {round(conf*100)}%"
        note += f" {obj}:{round(conf*100)}%"
        # Figure out and draw a background for the label
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(
            image,
            (top_left[0], top_left[1] - text_h - (padding * 2)),
            (top_left[0] + text_w + (padding * 2), top_left[1]),
            frame_color,
            -1
        )

        # Draw a bounding box and label.
        cv2.rectangle(image, top_left, bottom_right, frame_color, 2)
        cv2.putText(image, label, (top_left[0] + padding, top_left[1] - padding), font, font_scale,
                    (255, 255, 255), font_thickness)

    cv2.imwrite(image_path, image)
    logger.info("Image saved")

    with open(json_path, 'w') as jf:
        json.dump(detect_info, jf)

    # Update notes
    note += '</a>'
    return note

def main():
    init()
    last_detect = datetime.min
    print("Beginning monitoring loop")
    cap = VideoCapture()

    frame_count = 0
    frame_time = time.time()
    while True:
        t1 = time.time()
        try:
            ret, snapshot = cap.read(timeout=5)

            # Convert to a PIL image for faster resizing
            snapshot = Image.fromarray(snapshot)
        except Exception:
            # If we get some sort of an error, log it, wait 5 seconds, then try again
            time.sleep(5)
            continue

        matched, objects = process_image(snapshot)
        logger.debug(f"{matched}, {objects}")

        if matched:
            logger.info("Matched object. Signaling monitor.")
            try:
                requests.get(config['action']['action_url'])
            except Exception:
                logger.exception("Unable to call video")

            mqtt_broker = config['action'].get('mqtt_broker', None)
            if mqtt_broker:
                mqtt_user = config['action'].get('mqtt_user', None)
                mqtt_password = config['action'].get('mqtt_password', None)
                auth = None
                if mqtt_user and mqtt_password:
                    auth={'username':mqtt_user, 'password':mqtt_password}
                try:
                    publish_mqtt.single('cameramon/object', payload='detected',
                                   hostname=mqtt_broker,
                                   client_id="cameramon",
                                   auth=auth)
                except socket.timeout:
                    logger.warning("Unable to post MQTT messge: Timeout")

            if datetime.now() - last_detect > timedelta(seconds = 20):
                # Only save a new image if it has been more than 20 seconds
                # since the last object Detected.
                # Otherwise, it's probably the same object, just moved,
                # so no need for a new image.
                save_image(objects, snapshot)
            else:
                logger.info("Not saving image due to recent detection")

            last_detect = datetime.now()

        logger.debug(f"Ran detect loop in {time.time() - t1}")

        frame_count += 1
        elapsed_time = time.time() - frame_time
        if elapsed_time > 10:
            fps = frame_count / elapsed_time
            logger.info(f"FPS: {fps:.2f}")
            frame_count = 0
            frame_time = time.time()

        # limit prcessing to 10 FPS
        if time.time() - t1 < .1:
            time.sleep(.1 - (time.time() - t1))

    print("Exiting detection loop and ending process")
