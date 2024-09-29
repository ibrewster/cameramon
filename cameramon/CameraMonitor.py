import configparser
import io
import logging
import math
import os
import re
import socket
import threading
import time


from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from logging import handlers

import cv2
import numpy
import requests
import ujson as json

import paho.mqtt.publish as publish_mqtt
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

coral_tpu = False

try:    
    from pycoral.adapters.common import input_size, input_tensor
    from pycoral.adapters.detect import get_objects
    from pycoral.utils.dataset import read_label_file
    from pycoral.utils.edgetpu import make_interpreter
    coral_tpu = True
except ImportError:
    import tensorflow as tf
    
    def read_label_file(label_file):
        """Reads a label file and returns a dictionary mapping label indices to label names."""
        with open(label_file, 'r') as f:
            labels = {}
            for line in f.readlines():
                i, label = line.strip().split(' ', 1)
                labels[int(i)] = label.strip()
            return labels
        
    def input_size(interpreter):
        """Returns the input size for the model."""
        input_details = interpreter.get_input_details()
        return input_details[0]['shape'][1:3]  # Assuming the input shape is [1, height, width, channels]
    
    def input_tensor(interpreter):
        """Returns the input tensor for the model."""
        input_details = interpreter.get_input_details()
        input_index = input_details[0]['index']
        return interpreter.get_tensor(input_index)[0]
    
    class DetectedObject:
        bbox = None
        id = None
        score = None
        
    def get_objects(interpreter, min_confidence):
        """Extracts detected objects from the output tensor."""
        output_details = interpreter.get_output_details()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class indices
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
    
        objs = []
        for i in range(len(scores)):
            if scores[i] >= min_confidence:
                obj = DetectedObject()
                bbox = boxes[i]
                bbox = numpy.array([
                    bbox[1],  # xmin
                    bbox[0],  # ymin
                    bbox[3],  # xmax
                    bbox[2]   # ymax
                ]) * 300                
                
                obj.bbox = bbox.tolist()
                obj.id = int(classes[i])
                obj.score = scores[i]
                objs.append(obj)
        return objs

print("My PID is:", os.getpid())

conf_ini = configparser.ConfigParser()
MODULE_PATH = os.path.dirname(__file__)
conf_ini.read(os.path.join(MODULE_PATH, 'cameramon.ini'))

interpreter = None
inference_size = None
labels = None
zones = None
car_zones = None
logger = None
trackers = defaultdict(list)
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
    try:
        interpreter = make_interpreter(conf_ini['model']['object_weights'])
    except:
        interpreter = tf.lite.Interpreter(conf_ini['model']['object_weights'])

    interpreter.allocate_tensors()
    labels = read_label_file(conf_ini['model']['object_labels'])
    inference_size = input_size(interpreter)

    logger.info("Loading Zones")
    zones, car_zones = load_zones(3)

def init_logging():
    global logger
    FORMAT = "%(asctime)-15s %(levelname)s: %(message)s"
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)

    # File logging
    handler = handlers.RotatingFileHandler('/var/log/cameramon/cameramon.log',
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
    if coral_tpu:
        # Direct data access
        input_tensor(interpreter)[:h, :w] = image
    else:
        # Tensorflow CPU
        tensor = input_tensor(interpreter)
        tensor[:h, :w] = image
    
        input_details = interpreter.get_input_details()
        
        tensor = numpy.expand_dims(tensor, axis=0)
        interpreter.set_tensor(input_details[0]['index'], tensor)
    
    interpreter.invoke()

    objs = get_objects(interpreter, conf_ini['model']['min_conf'])
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
                # object hasn't been seen in the past second. Remove tracker
                # Detection runs at something like 10 fps, so if not touched
                # in a full second, that's 10 detections in a row where it 
                # hasn't matched anything.
                logger.warning("Stale tracker detected. Removing!")
                trackers[class_id].pop(i)
                continue
                
            success, updated_bbox = tracker.update(frame)

            if not success:
                logger.warning("Object gone. Removing from list")
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

clear_frames = 0
def detect_image(image, img_ratio, img_area):
    global clear_frames
    
    update_trackers(image)

    logger.debug("Processing image for event")
    found_match = False
    canidate_objects = []
    
    objs = run_inference(image)
    # print(objs)

    if not objs:
        if clear_frames < 100:
            clear_frames += 1
        return (found_match, canidate_objects)
    else:
        clear_frames = 0

    new_detections = defaultdict(list)

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
        detected_bbox = box(*item.bbox)

        poly_area = bbox_poly.area
        if poly_area / img_area > conf_ini['match']['max_det_size']:
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

        # Good object, in our zones.
        # Store as a match
        new_detections[obj].append(bbox_poly)

        # And list as a canidate for alerting
        canidate_objects.append((obj, bbox, conf))

        if obj in vehicles:
            # Compare to *any* vehicles we have seen, as it often gets confused
            comp_objects = (shape
                            for vehicle in vehicles
                            for shape in trackers.get(vehicle, []))
        else:
            comp_objects = trackers.get(obj, []).copy()
        
        if not comp_objects:
            max_iou = -1
        else:
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
            
        if max_iou > 0.2:
            tracker.touch()
            if tracker.is_moving:  # Define your velocity threshold
                logger.warning(f"Object {obj} has moved.")
                found_match = True  # The object has moved                
            break
        else:
            # IoU indicates no match, therfore new object.
            logger.warning(f"Matching {obj} as it appears to be new. Max iou: {max_iou}")
            found_match = True
            new_tracker = Tracker()
            new_tracker.init(image, detected_bbox)
            trackers[obj].append(new_tracker)
            

    return (found_match, canidate_objects)


def process_image(pil_image):
    if pil_image is None or not any(pil_image.size):
        raise TypeError("Image was not an image")

    # t1 = time.time()
    resized_pil,target_ratio,img_area = resize_pil_image(pil_image)
    # print(f"Resized image to {inference_size} in {time.time() - t1}")
    
    opencv_image = numpy.asarray(resized_pil)

    found_match, all_objects = detect_image(opencv_image, target_ratio, img_area)

    return (bool(found_match), all_objects)


def resize_pil_image(pil_image): 
    target_ratio = max(pil_image.size[0] / inference_size[0],
                       pil_image.size[1] / inference_size[1])

    img_area = pil_image.size[0] * pil_image.size[1]
    logger.debug("Resizing image")

    #t1 = time.time()
    new_size = (numpy.asarray(pil_image.size) // target_ratio).astype(int)
    resized_pil = pil_image.resize(new_size.tolist(), reducing_gap=2.0)
    return (resized_pil, target_ratio, img_area)


def process_opencv_image(original_image):
    if original_image is None or not any(original_image.shape):
        raise TypeError("Image was not an image")

    target_ratio = max(original_image.shape[1] / inference_size[0],
                       original_image.shape[0] / inference_size[1])

    img_area = original_image.shape[0] * original_image.shape[1]
    logger.debug("Resizing image")

    # t1 = time.time()
    new_size = (numpy.asarray(original_image.shape[:-1]) // target_ratio).astype(int)
    opencv_image = cv2.resize(original_image, new_size, interpolation = cv2.INTER_AREA)
    # print(f"Resized image to {inference_size} in {time.time() - t1}")

    found_match, all_objects, match_info = detect_image(opencv_image, target_ratio, img_area)

    return (bool(found_match), all_objects, match_info, original_image)


def save_image(objects, image):
    # convert image to an openCV image for editing
    if not isinstance(image, numpy.ndarray):
        # No need to convert color channels., Since PIL was only used 
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

t0 = None
def init_video_stream():
    global t0
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"    
    url = 'rtsp://watchman.brewstersoft.net:8554/drivecam?fflags=nobuffer'
    # url = 'http://localhost/zm/cgi-bin/nph-zms?monitor=3&scale=100&maxfps=30&buffer=1000&user=israel&pass=shanima81'
    cap = cv2.VideoCapture(url)
    backend_name = cap.getBackendName()
    print(f"Using backend: {backend_name}")
    if backend_name == "GSTREAMER":
        # Use a GStreamer pipeline to control buffering
        pipeline = f"rtspsrc location={url} latency=0 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true"
        cap.release()  # Release the previous VideoCapture
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        print("Configured GStreamer pipeline.")    
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.grab()
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    t0 = (time.time() * 1000) - timestamp
    logger.warning(f"Timestamp at first is: {timestamp}")    
    return cap

class VideoCapture:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self._frame_available = threading.Event()
        self._capture = init_video_stream()
        self.running = True        
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        
    def update(self):
        while self.running:
            try:
                ret, frame = self._capture.read()
                if not ret:
                    raise EOFError("Nothing to read")
            except Exception:
                logger.exception("Unable to retrieve zoneminder image")
                self._capture.release()
                time.sleep(5)
                self._capture = init_video_stream()
                logger.warning("Trying again...")
                continue
            
            current_time = time.time() * 1000   # Convert current time to ms
            delay = current_time - (t0 + self._capture.get(cv2.CAP_PROP_POS_MSEC))
            if delay > 500:
                # Reinitialize the video stream
                logger.warning("Large delay detected. Re-initializing.")
                self._capture.release()
                self._capture = init_video_stream()
                self._capture.grab()
                continue
            if delay > 300:
                self._capture.grab()
            # Yes, this logic means both will run if the delay is greater than 200.
            if delay > 200:
                self._capture.grab()
                logger.warning(f"The current delay is: {delay}")
            
            with self.lock:
                self.frame = frame
                self._frame_available.set()
                
    def read(self, timeout=None):
        try:
            self._frame_available.wait(timeout)
        except TimeoutError:
            return (None, None)

        with self.lock:
            frame = (True, self.frame.copy()) if self.frame is not None else (False, None)
            self._frame_available.clear()
            return frame
        
    def stop(self):
        self.running = False
        self._capture.release()    
  
@lru_cache(maxsize=128)
def get_diagonal_length(bbox: Polygon) -> float:
    minx, miny, maxx, maxy = bbox.bounds
    return math.dist((minx, miny), (maxx, maxy))

class Tracker:
    THRESHOLD_PERCENT = 0.15
    KEYFRAME_INTERVAL = 60
    
    # params = cv2.TrackerKCF_Params()
    # params.sigma = 0.3
    # params.lambda_ = 0.0001
    
    def __init__(self):
        # self._tracker = cv2.TrackerKCF_create(self.params)
        self._tracker = cv2.TrackerCSRT_create()
        self.is_moving = False
        self._ref_bbox = None
        self._bbox = None
        self.last_seen = None
        self._last_update = None
        self._update_count = 0
        self._motion_sum = 0
        self._max_move = 0
        self._last_ref_frame_time = None
        
        
    def init(self, image, bounding_box: box):
        minx, miny, maxx, maxy = bounding_box.bounds
        width = int(maxx - minx)
        height = int(maxy - miny)
        bbox = (int(minx), int(miny), width, height)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        self._tracker.init(image, bbox)
        
        self._bbox = bounding_box
        self.last_seen = time.time()
        self._last_update = time.time()
    
    def update(self, frame):
        if time.time() - self._last_update < 0.1:
            return (True, self._bbox)
        
        self._last_update = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        success, updated_bbox = self._tracker.update(frame)
        if success:
            x, y, w, h = updated_bbox
            shapely_updated_bbox = box(x, y, x + w, y + h)
            self._bbox = shapely_updated_bbox
            if self._ref_bbox is None:
                self._ref_bbox = shapely_updated_bbox
                self._last_ref_frame_time = time.time()
        else:
            self._bbox = None

        self._calc_motion()
        return success, self._bbox
    
    def _calc_motion(self):
        if self._bbox is None:
            self.is_moving = False
            return

        ref_len = max(get_diagonal_length(self._ref_bbox),
                      get_diagonal_length(self._bbox))
        
        dist_moved = self._ref_bbox.centroid.distance(self._bbox.centroid)
        percent_moved = dist_moved / ref_len
        
        self._motion_sum += percent_moved
        self._update_count += 1
        self._max_move = max(self._max_move, percent_moved)
        avg_movement = (self._motion_sum / self._update_count) * 100        

        # DEBUGING CODE - CHANGE TO DEBUG LEVEL OR REMOVE IN PRODUCTION
        if self._update_count % 120 == 0:
            logger.warning(f"Avg Movement {avg_movement:.2f}% Max: {self._max_move * 100:.2f}")
        #  END DEBUGGING CODE
            
        self.is_moving = percent_moved > self.THRESHOLD_PERCENT
            
        if self.is_moving or (
            time.time() - self._last_ref_frame_time > self.KEYFRAME_INTERVAL
            and avg_movement > 1 # percent
            ):
            if self.is_moving:
                logger.warning(f"Object moved {percent_moved * 100:.2f}%")            
            logger.warning("Updating reference frame")
            self._ref_bbox = self._bbox
            self._last_ref_frame_time = time.time()
                
            # Also reset average and max for good measure
            self._max_move = self._motion_sum = 0
            self._update_count = 0                
    
    def touch(self):
        self.last_seen = time.time()
        
    @property
    def bbox(self):
        return self._bbox

if __name__ == "__main__":
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
            
            # rgb_image = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
            # Convert to a PIL image for faster resizing
            snapshot = Image.fromarray(snapshot)
            # snapshot = download_zm_image()
        except Exception:
            # If we get some sort of an error, log it, wait 5 seconds, then try again
            time.sleep(5)
            continue

        matched, objects = process_image(snapshot)
        # matched, objects, matched_obj, image = process_opencv_image(snapshot)
        # print(matched, objects, matched_obj)
        
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
                
            mqtt_broker = conf_ini['action'].get('mqtt_broker', None)
            if mqtt_broker:
                mqtt_user = conf_ini['action'].get('mqtt_user', None)
                mqtt_password = conf_ini['action'].get('mqtt_password', None)
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
                note = save_image(objects, snapshot)
            else:
                logger.info("Not saving image due to recent detection")

            last_detect = datetime.now()

        logger.debug(f"Ran detect loop in {time.time() - t1}")
        # print("Ran detect loop in", time.time() - t1)

        frame_count += 1
        elapsed_time = time.time() - frame_time
        if elapsed_time > 10:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            frame_time = time.time()
            
        # limit prcessing to 10 FPS
        if time.time() - t1 < .1:
            time.sleep(.1 - (time.time() - t1))

    print("Exiting detection loop and ending process")
