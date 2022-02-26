import configparser
import logging
import os
import re
import time

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from logging import handlers

import cv2
import imutils
import numpy
import requests
import ujson as json

import pyzm.ml.object as object_detection

from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

# Pull image directly from camera. For now, use zoneminder images to reduce load on camera
# def download_camera_image():
#     auth_user = 'admin'
#     auth_password = 'shanima81'
#     auth = requests.auth.HTTPBasicAuth(auth_user, auth_password)
#     url = 'http://10.27.81.62/tmpfs/auto.jpg'
#     res = requests.get(url, auth=auth)
#
#     if res.status_code != 200:
#         print("Error when retrieving file!")
#         raise requests.exceptions.HTTPError("Bad Content")
#
#     snapshot_img = numpy.frombuffer(res.content,
#                                     numpy.uint8)
#
#     return snapshot_img


def download_zm_image():
    url1 = 'http://localhost/zm/cgi-bin/nph-zms'
    args = {
        'mode': 'single',
        'monitor': '3',
        'scale': 100,
        'maxfps': 5,
        'buffer': 1000,
        'user': 'israel',
        'pass': 'shanima81',
    }

    res = requests.get(url1, args)

    if res.status_code != 200:
        logger.warning("Error when retrieving file!")
        raise requests.exceptions.HTTPError("Bad Content")

    snapshot_img = numpy.frombuffer(res.content,
                                    numpy.uint8)

    return snapshot_img


config = {
    'object_framework': 'coral_edgetpu',
    'object_min_confidence': .345,
    'tpu_max_processes': 1,
    'object_weights': '/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
    # 'object_weights': '/var/lib/zmeventnotification/models/coral_edgetpu/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite',
    'object_labels': '/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names',
    'disable_locks': 'yes',
    'past_det_max_diff_area': .08,  # 8%
    'max_detection_size': '80%',  # 90%
    'image_directory': '/data/zoneminder',
    'api_portal': 'https://watchman.brewstersoft.net/zm/api',
}

conf_ini = configparser.ConfigParser()
conf_ini.read('/etc/zm/zmdetect.ini')

model = None
zones = None
logger = None
prev_detections = defaultdict(list)
detect_pattern = re.compile('(person|car|motorbike|bus|truck|boat|skateboard|horse|dog|cat)')


def init():
    global model
    global zones

    init_logging()

    logger.info("Setting up model")
    model = object_detection.Object(options=config)
    zones = load_zones(3)


# Stupid overides since pyzm doesn't use the default logging module
def zmlog_Debug(level, message, caller = None):
    return logger.debug(message)


def zmlog_Info(message, caller = None):
    return logger.info(message)


def zmlog_Warning(message, caller = None):
    return logger.warning(message)


def zmlog_Error(message, caller = None):
    return logger.error(message)


def zmlog_Fatal(message, caller = None):
    return logger.fatal(message)


def init_logging():
    global logger
    FORMAT = "%(asctime)-15s %(levelname)s: %(message)s"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    # File logging
    handler = handlers.RotatingFileHandler('/var/log/drive_monitor.log',
                                           maxBytes=1024000, backupCount=5)
    fmt = logging.Formatter(FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(fmt)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)

    logging.basicConfig(format=FORMAT, level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')

    logger.Debug = zmlog_Debug
    logger.Info = zmlog_Info
    logger.Warning = zmlog_Warning
    logger.Error = zmlog_Error
    logger.Fatal = zmlog_Fatal
    logger.Panic = zmlog_Fatal

    object_detection.g.logger = logger


def process_coordinates(coords):
    points = coords.split()
    points = [(float(y), float(z)) for y, z in [x.split(',') for x in points]]
    poly = Polygon(points)
    return poly


def load_zones(monitor):
    # Get a list of zones for our monitor
    logger.info(f"Loading zones for monitor {monitor}")
    api_user = conf_ini['general']['zm_api_user']
    api_password = conf_ini['general']['zm_api_password']
    auth = {'user': api_user,
            'pass': api_password, }
    zm_url = conf_ini['general']['zm_portal_url']
    api_url = f"{zm_url}/api/zones/forMonitor/{monitor}.json"
    resp = requests.get(api_url, auth)
    zone_list = resp.json()
    zone_list = zone_list['zones']
    zone_coords = [process_coordinates(x['Zone']['Coords']) for x in
                   zone_list if x['Zone']['Type'] == 'Active']

    logger.debug(str(zone_coords))
    zones = prep(unary_union(zone_coords))
    logger.debug(str(zones))
    return zones


def detect_image(image, img_ratio):
    global prev_detections

    logger.info("Processing image for event")
    match = None
    found_match = False

    new_detections = defaultdict(list)
    past_det_threshold = 1 - config['past_det_max_diff_area']

    if image is None:
        return False  # No image means no object to detect

    logger.info(f"Detecting objects in image")

    bboxes, objects, confs, method = model.detect(image)
    scaled_bboxes = numpy.array(bboxes) * img_ratio
    canidate_objects = []

    logger.info("Object detection complete. Processing results")

    for bbox, obj, conf in zip(scaled_bboxes, objects, confs):
        if conf < config['object_min_confidence']:
            # logger.info(f"Ignoring {obj} {bbox} as conf. level {conf} is lower than {config['object_min_confidence']}")
            continue  # We don't trust this object, so move on

        if not detect_pattern.match(obj):
            # logger.info(f"Ignoring {obj} {bbox} as it does not match our detect patern")
            continue  # Not an object of interest to us

        bbox_poly = box(*bbox)

#         bbox_area = bbox_poly.area
#         bbox_percent = bbox_area / img_area
#         logger.debug(f"Detect Percent: {bbox_percent}")
#         if bbox_percent > config.get('max_detection_size', .9):
#             logger.info(f"Ignoring {obj} {bbox} as object area exceeds max detection size ({bbox_percent})")
#             continue

        if not zones.intersects(bbox_poly):
            logger.info(f"Ignoring {obj} {bbox_poly.bounds} as it is outside our zones")
            continue

        # Good object, in our zones.
        # Store as a match
        new_detections[obj].append(bbox_poly)

        # And list as a canidate for alerting
        canidate_objects.append((obj, bbox, conf))

        # See if this is a *new* match
        if found_match:
            # Don't bother checking if this is a repeat,
            # we already found something that wasn't.
            # Just store this match (above) and move on.
            continue

        comp_objects = prev_detections[obj]
        # Cars and trucks are sometimes confused, so look for either when one is found
        if obj == "car":
            comp_objects += prev_detections.get('truck', [])

        if obj == "truck":
            comp_objects += prev_detections.get('car', [])

        for past_match in comp_objects:
            intersect_area = past_match.intersection(bbox_poly).area
            bbox_percent = intersect_area / bbox_poly.area
            past_percent = intersect_area / past_match.area

            logger.debug(f"bbox intersect area for object {obj}, {conf}: {bbox_percent} past percent: {past_percent}")

            if intersect_area != 0 and \
               (bbox_percent > past_det_threshold
                    and past_percent > past_det_threshold):
                logger.info(f"Ignoring {obj} {bbox} as we have already seen it")
                break
        else:
            # Didn't break out of for loop, therfore object didn't match
            # *anything* in past objects
            # This means it is a new, or moved, object.
            logger.debug(f"Matching {obj} as it appears to be new or moved")
            found_match = True
            match = (obj, bbox_poly.bounds, conf)

    if found_match:
        prev_detections = new_detections
        obj, bbox_poly, conf = match
        logger.info(f"Matched {obj} {bbox_poly} with conf. level {conf}")
        logger.debug(f"All Objects: {list(zip(objects, confs, bboxes))}")

    return (found_match, canidate_objects, match)


def process_image(raw_image):
    detect_size = 500

    if raw_image is not None and raw_image.size > 0:
        # logger.info("Loading camera image")
        image = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)
        img_ratio = image.shape[1] / detect_size
        resized_image = imutils.resize(image, width = detect_size)

    else:
        # logger.warning("No image retreived to analyze")
        raise TypeError("Image was not an image")

    if resized_image is None:
        # logger.warning("No image retreived to analyze")
        raise TypeError("Image was not an image")

    found_match, all_objects, match_info = detect_image(resized_image, img_ratio)

    return (bool(found_match), all_objects, match_info, image)


def save_image(objects, match, image):
    eventid = 1
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = .75
    font_thickness = 1
    frame_color = (46, 174, 85)
    padding = 3
    img_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    file_path = os.path.join('/data/detect_images', '3',
                             img_date)

    os.makedirs(file_path, exist_ok = True)

    image_path = os.path.join(file_path, 'objdetect.jpg')
    json_path = os.path.join(file_path, 'objects.json')

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
        detect_info['confidences'].append(conf)
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

    with open(json_path, 'w') as jf:
        json.dump(detect_info, jf)

    # Update notes
    note += '</a>'
    return note


if __name__ == "__main__":
    init()
    # Get the inital snapshot
    snapshot = download_zm_image()
    with ThreadPoolExecutor(max_workers = 2) as executor:
        future_snap = None
        while True:
            t1 = time.time()
            if future_snap is not None:
                try:
                    snapshot = future_snap.result()
                except requests.exceptions.HTTPError:
                    # If we get an error when trying to get an image, wait 5 seconds then try again.
                    logger.exception("Error when retrieving image. Trying again in 5 seconds")
                    time.sleep(5)
                    future_snap = executor.submit(download_zm_image)
                    continue

            future_snap = executor.submit(download_zm_image)
            matched, objects, matched_obj, image = process_image(snapshot)

            if matched:
                try:
                    requests.get(conf_ini['action']['action_url'])
                except Exception:
                    logger.exception("Unable to call video")
                    pass

                note = save_image(objects, matched_obj, image)

            # limit prcessing to 5 FPS
            if time.time() - t1 < .2:
                time.sleep(.2 - (time.time() - t1))

            logger.info(f"Ran detect loop in {time.time() - t1}")
