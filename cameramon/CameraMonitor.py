import configparser
import io
import logging
import os
import re
import time

from collections import defaultdict
from datetime import datetime, timedelta
from logging import handlers

import cv2
import numpy
import requests
import ujson as json

from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

from pycoral.adapters.common import input_size, input_tensor
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def download_camera_image():
    """Code to pull image directly from camera rather than via zoneminder"""
    auth_user = 'admin'
    auth_password = 'shanima81'
    auth = requests.auth.HTTPBasicAuth(auth_user, auth_password)
    url = 'http://10.27.81.62/tmpfs/auto.jpg'
    res = requests.get(url, auth=auth)
#
    if res.status_code != 200:
        print("Error when retrieving file!")
        raise requests.exceptions.HTTPError("Bad Content")
#
    # snapshot_img = numpy.frombuffer(res.content,
    #                                numpy.uint8)

    snapshot_img = Image.open(io.BytesIO(res.content))

    return snapshot_img


ZM_IMG_URL = 'https://watchman.brewstersoft.net/zm/cgi-bin/nph-zms'
ZM_IMG_ARGS = {
    'mode': 'single',
    'monitor': '3',
    'scale': 100,
    'maxfps': 15,
    'buffer': 1000,
    'user': 'israel',
    'pass': 'shanima81',
}


def download_zm_image(fmt = 'pil'):
    res = requests.get(ZM_IMG_URL, ZM_IMG_ARGS)

    if res.status_code != 200:
        logger.warning("Error when retrieving file!")
        raise requests.exceptions.HTTPError("Bad Content")

    if fmt == 'pil':
        snapshot_img = Image.open(io.BytesIO(res.content))
    elif fmt == 'opencv':
        numpy_img = numpy.frombuffer(res.content, numpy.uint8)
        snapshot_img = cv2.imdecode(numpy_img, cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"Invalid image format specified ({fmt}). Must be one of: ['pil', 'opencv']")
    
    return snapshot_img


print("My PID is:", os.getpid())
config = {
    'object_framework': 'coral_edgetpu',
    'object_min_confidence': .345,
    # 'object_min_confidence': .45,
    'tpu_max_processes': 1,
    'object_weights': '/usr/local/cameramon/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
    # 'object_weights': '/usr/local/cameramon/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite',
    # 'object_weights': '/usr/local/cameramon/models/efficientdet_lite3x_640_ptq_edgetpu.tflite',
    # 'object_weights': '/usr/local/cameramon/models/efficientdet_lite3_512_ptq_edgetpu.tflite',
    # 'object_weights': '/usr/local/cameramon/models/efficientdet_lite1_384_ptq_edgetpu.tflite',
    # 'object_weights': '/usr/local/cameramon/models/efficientdet_lite2_448_ptq_edgetpu.tflite',
    'object_labels': '/usr/local/cameramon/models/coco_indexed.names',
    'past_det_max_diff_area': .20,  # 20%
    'max_detection_size': .3,  # 30%
    'api_portal': 'https://watchman.brewstersoft.net/zm/api',
}

conf_ini = configparser.ConfigParser()
MODULE_PATH = os.path.dirname(__file__)
conf_ini.read(os.path.join(MODULE_PATH, 'cameramon.ini'))

interpreter = None
inference_size = None
labels = None
zones = None
car_zones = None
logger = None
prev_detections = defaultdict(list)
detect_pattern = re.compile('(person|car|motorbike|bus|truck|boat|skateboard|horse|dog|cat)')
interesting_objects = ("person", "bicycle", "car", "motorbike", "bus", "truck", "boat", "skateboard", "horse", "dog", "cat")
LOG_LEVEL = logging.INFO


def init():
    global interpreter
    global labels
    global inference_size
    global zones
    global car_zones

    init_logging()

    logger.info("Setting up model")
    interpreter = make_interpreter(config['object_weights'])
    interpreter.allocate_tensors()
    labels = read_label_file(config['object_labels'])
    inference_size = input_size(interpreter)

    logger.info("Loading Zones")
    zones, car_zones = load_zones(3)


def init_logging():
    global logger
    FORMAT = "%(asctime)-15s %(levelname)s: %(message)s"
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)

    # File logging
    handler = handlers.RotatingFileHandler('/var/log/cameramon.log',
                                           maxBytes=1024000, backupCount=5)
    fmt = logging.Formatter(FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(fmt)
    handler.setLevel(LOG_LEVEL)
    logger.addHandler(handler)

    logging.basicConfig(format=FORMAT, level=LOG_LEVEL, datefmt='%Y-%m-%d %H:%M:%S')

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
    
    no_car_names = ['Yard', 'Front Bush1', 'Side Bushes', 'Porch']
    car_zone_coords = [
        process_coordinates(x['Zone']['Coords'])
        for x in zone_list
        if x['Zone']['Type'] == 'Active'
        and x['Zone']['Name'] not in no_car_names]
    
    car_zone = unary_union(car_zone_coords)
    logger.debug(str(car_zone))
    car_zones = prep(car_zone)
    
    return (zones, car_zones)

def run_inference(image):
    logger.debug(f"Detecting objects in image")

    h, w, channel = image.shape
    input_tensor(interpreter)[:h, :w] = image
    interpreter.invoke()

    objs = get_objects(interpreter, config['object_min_confidence'])
    return objs

def detect_image(image, img_ratio, img_area):
    global prev_detections

    logger.debug("Processing image for event")
    match = None
    found_match = False
    canidate_objects = []
    
    objs = run_inference(image)
    # print(objs)

    if not objs:
        return (found_match, canidate_objects, match)

    new_detections = defaultdict(list)
    past_det_threshold = 1 - config['past_det_max_diff_area']

    logger.debug("Object detection complete. Processing results")

    objects = []
    confs = []
    bboxes = []
    for item in objs:
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

        poly_area = bbox_poly.area
        if poly_area / img_area > config['max_detection_size']:
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
        if obj in vehicles and bbox[1] < 20 and bbox[2] > (full_width - 30):
            logger.info(f"Got car in yard (bbox: {bbox}, image width: {full_width}). Ignoring.")
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

        if obj in vehicles:
            # Compare to *any* vehicles we have seen, as it often gets confused
            comp_objects = (shape
                            for vehicle in vehicles
                            for shape in prev_detections.get(vehicle, []))
        else:
            comp_objects = prev_detections.get(obj, []).copy()

        for past_match in comp_objects:
            intersect_area = past_match.intersection(bbox_poly).area
            bbox_percent = intersect_area / bbox_poly.area
            past_percent = intersect_area / past_match.area

            logger.debug(f"bbox intersect area for object {obj}, {round(conf,2)}: {round(bbox_percent,2)} past percent: {round(past_percent,2)}")

            if intersect_area != 0 and \
               (bbox_percent > past_det_threshold and
                    past_percent > past_det_threshold):
                logger.debug(f"Ignoring {obj} {bbox} as we have already seen it")
                break
            # else:
                # logger.info(f"bbox intersect area for object {obj} with conf {round(conf,2)}: {round(bbox_percent,2)}/{round(past_percent,2)}")
        else:
            # Didn't break out of for loop, therfore object didn't match
            # *anything* in past objects
            # This means it is a new, or moved, object.
            logger.debug(f"Matching {obj} as it appears to be new or moved")
            found_match = True
            match = (obj, bbox_poly.bounds, conf)

    if found_match:
        prev_detections = new_detections
        print("***********Replaced prev_detections with new_detections")
        obj, bbox_poly, conf = match
        logger.info(f"Matched {obj} {bbox_poly} with conf. level {conf}")
        logger.debug(f"All Objects: {list(zip(objects, confs, bboxes))}") 
    
    # for key, value in prev_detections.items():
        # print(f"Stored {len(value)} objects of type {key}")
    return (found_match, canidate_objects, match)


def process_image(pil_image):
    if pil_image is None or not any(pil_image.size):
        raise TypeError("Image was not an image")

    #print(f"Resized image to {inference_size} in {time.time() - t1}")
    resized_pil,target_ratio,img_area = resize_pil_image(pil_image)

    opencv_image = numpy.asarray(resized_pil)

    found_match, all_objects, match_info = detect_image(opencv_image, target_ratio, img_area)

    return (bool(found_match), all_objects, match_info, pil_image)


def resize_pil_image(pil_image): 
    target_ratio = max(pil_image.size[0] / inference_size[0],
                       pil_image.size[1] / inference_size[1])

    img_area = pil_image.size[0] * pil_image.size[1]
    logger.debug("Resizing image")

    #t1 = time.time()
    new_size = (numpy.asarray(pil_image.size) // target_ratio).astype(int)
    resized_pil = pil_image.resize(new_size, reducing_gap=2.0)
    return (resized_pil, target_ratio, img_area)


def process_opencv_image(original_image):
    if original_image is None or not any(original_image.shape):
        raise TypeError("Image was not an image")

    target_ratio = max(original_image.shape[1] / inference_size[0],
                       original_image.shape[0] / inference_size[1])

    img_area = original_image.shape[0] * original_image.shape[1]
    logger.debug("Resizing image")

    #t1 = time.time()
    new_size = (numpy.asarray(original_image.shape[:-1]) // target_ratio).astype(int)
    opencv_image = cv2.resize(original_image, new_size, interpolation = cv2.INTER_AREA)
    #print(f"Resized image to {inference_size} in {time.time() - t1}")

    found_match, all_objects, match_info = detect_image(opencv_image, target_ratio, img_area)

    return (bool(found_match), all_objects, match_info, original_image)


def save_image(objects, match, image):
    # convert image to an openCV image for editing
    if not isinstance(image, numpy.ndarray):
        image = numpy.asarray(image)[:, :, ::-1].copy()
    
    eventid = 1
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = .75
    font_thickness = 1
    frame_color = (46, 174, 85)
    padding = 3
    img_date = datetime.now().strftime('%Y-%m-%d')
    img_time = datetime.now().strftime('%H-%M-%S')

    file_path = os.path.join(conf_ini['action']['image_directory'], 'Driveway',
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
    logger.info("Image saved")

    with open(json_path, 'w') as jf:
        json.dump(detect_info, jf)

    # Update notes
    note += '</a>'
    return note

def init_zm_stream():
    url = 'http://localhost/zm/cgi-bin/nph-zms?monitor=3&scale=100&maxfps=30&buffer=1000&user=israel&pass=shanima81'
    cap = cv2.VideoCapture(url)
    return cap

if __name__ == "__main__":
    init()
    last_detect = datetime.min
    print("Beginning monitoring loop")
    while True:
        t1 = time.time()
        try:
            snapshot = download_zm_image()
        except Exception:
            # If we get some sort of an error, log it, wait 5 seconds, then try again
            logger.exception("Unable to retrieve zoneminder image")
            time.sleep(5)
            logger.warning("Trying again...")
            continue

        matched, objects, matched_obj, image = process_image(snapshot)
        #matched, objects, matched_obj, image = process_opencv_image(snapshot)

        if matched:
            # Check another image, pulled directly from the camera
            # logger.info("Matched object. Double-checking.")
            # t_check = time.time()
            # snapshot = download_camera_image()
            # confirm, _, _, _ = process_image(snapshot)
            # logger.info(f"Double-checked in {time.time() - t_check} with result {confirm}")
            
            # if confirm:
                logger.info("Matched object. Signaling monitor.")
                try:
                    requests.get(conf_ini['action']['action_url'])
                except Exception:
                    logger.exception("Unable to call video")
    
                if datetime.now() - last_detect > timedelta(seconds = 20):
                    # Only save a new image if it has been more than 20 seconds
                    # since the last object Detected.
                    # Otherwise, it's probably the same object, just moved,
                    # so no need for a new image.
                    note = save_image(objects, matched_obj, image)
                else:
                    logger.info("Not saving image due to recent detection")
    
                last_detect = datetime.now()
                time.sleep(2)  # Since we know there is motion/new object, we can wait
                # a couple of seconds before checking again.

        logger.debug(f"Ran detect loop in {time.time() - t1}")
        # print("Ran detect loop in", time.time() - t1)

        # limit prcessing to 10 FPS
        if time.time() - t1 < .1:
            time.sleep(.1 - (time.time() - t1))

    print("Exiting detection loop and ending process")
