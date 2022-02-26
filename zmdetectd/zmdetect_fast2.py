import configparser
import json
import logging
import os
import queue
import re
import time

from collections import defaultdict
from datetime import datetime
from logging import handlers
from urllib.error import HTTPError

from pyzm.ZMMemory import ZMMemory
import pyzm.ml.object as object_detection

import numpy
import requests
import cv2
import imutils
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

logger = None

# Global initalization. Yes, I use a lot of global variables here. So sue me.
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
mem3 = None


def init_logging():
    global logger
    FORMAT = "%(asctime)-15s %(levelname)s: %(message)s"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # File logging
    handler = handlers.RotatingFileHandler('/var/log/zoneminder/zmdetect.log',
                                           maxBytes=1024000, backupCount=5)
    fmt = logging.Formatter(FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(fmt)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def init():
    global model
    global mem3

    logger.info("Setting up model")
    model = object_detection.Object(options=config)

    logger.info("Connecting to shared memory")
    mem3 = ZMMemory(mid = 3)


portal_url = conf_ini['general']['zm_portal_url']
detect_pattern = re.compile(conf_ini['general']['detect_pattern'])

prev_detections = defaultdict(list)


def download_image():
    auth_user = conf_ini['frame']['image_user']
    auth_password = conf_ini['frame']['image_password']
    auth = requests.auth.HTTPBasicAuth(auth_user, auth_password)
    url = conf_ini['frame']['image_frame_url']
    res = requests.get(url, auth=auth)

    if res.status_code != 200:
        logger.error("Error when retrieving file!")
        raise HTTPError("Bad Content")

    snapshot_img = numpy.frombuffer(res.content,
                                    numpy.uint8)

    return snapshot_img


def process_event(event_id):
    logger.info(f"Processing event id {event_id}")

    # Make sure these are initalized even if download files fails
    raw_image = None
    # now download image(s)
    while True:
        try:
            logger.info("Getting image from camera")
            raw_image = download_image()
        except Exception as e:
            logger.exception(f'Error downloading files: {e}')

        image = None
        detect_size = 500

        if raw_image is not None and raw_image.size > 0:  # may be none
            logger.info("Loading camera image")
            image = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)
            img_ratio = image.shape[1] / detect_size
            resized_image = imutils.resize(image, width = detect_size)

        if resized_image is None:
            logger.warning("No image retreived to analyze. Trying again in a quarter second")
            time.sleep(0.25)
            continue

        found_match, all_objects, match_info = process_image(resized_image, img_ratio)

        if found_match:
            break

        data = mem3.get_shared_data()
        # Either we are no longer in alarm, or we have moved on to a new alarm. In either case, stop checking
        if data['state'] not in (1, 2, 3) or data['last_event'] != event_id:
            logger.info(f"Breaking because state is {data['state']} or event id {data['last_event']} != {event_id}")
            break

        # If we get here, then the event is still active, but we haven't seen anything
        logger.info("Nothing seen in image, but event still active.")
        time.sleep(.25)

    return (bool(found_match), all_objects, match_info, image)


def process_image(image, img_ratio):
    global prev_detections

    logging.info("Processing image for event")
    match = None
    found_match = False

    # old_detect = prev_detections
    # prev_detections = defaultdict(list)
    new_detections = defaultdict(list)
    past_det_threshold = 1 - config['past_det_max_diff_area']

    if image is None:
        return False  # No image means no object to detect

    logging.info(f"Detecting objects in image")

    bboxes, objects, confs, method = model.detect(image)
    scaled_bboxes = numpy.array(bboxes) * img_ratio
    canidate_objects = []

    logging.info("Object detection complete. Processing results")
#     img_area = reduce(mul, image.shape)

    for bbox, obj, conf in zip(scaled_bboxes, objects, confs):
        if conf < config['object_min_confidence']:
            logger.info(f"Ignoring {obj} {bbox} as conf. level {conf} is lower than {config['object_min_confidence']}")
            continue  # We don't trust this object, so move on

        if not detect_pattern.match(obj):
            logger.info(f"Ignoring {obj} {bbox} as it does not match our detect patern")
            continue  # Not an object of interest to us

        bbox_poly = box(*bbox)

#         bbox_area = bbox_poly.area
#         bbox_percent = bbox_area / img_area
#         logging.debug(f"Detect Percent: {bbox_percent}")
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


def save_image(objects, match, image, eventid):
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = .75
    font_thickness = 1
    file_path = os.path.join('/tmp/zmdetect', f"{eventid}.png")
    frame_color = (46, 174, 85)
    padding = 3
    img_date = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join(config['image_directory'], '3',
                             img_date, str(eventid))
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


def save_note(note, eventid):
    login_url = os.path.join(config['api_portal'], 'host', 'login.json')
    api_user = conf_ini['general']['zm_api_user']
    api_password = conf_ini['general']['zm_api_password']
    login = requests.post(login_url,
                          data = {'user': api_user,
                                  'pass': api_password,
                                  'stateful': '1',
                                  }
                          )

    logger.info(f"Logged on with result {login}")
    api_key = login.json()

    # Get the existing notes (if any)
    api_url = os.path.join(config['api_portal'], 'events', f"{eventid}.json")
    logger.info("Getting existing notes")
    try:
        resp = requests.get(api_url, params = {'token': api_key['access_token'], })
    except:
        logger.exception("Unable to get existing notes")
    old_note = ''
    if resp.status_code == 200:
        old_note = resp.json().get('event', {}).get('Event', {}).get('Notes', '')
        logger.info(f"Got existing note of {old_note}")
    else:
        logger.warning(f"Unable to get existing note {resp} {resp.text}")

    if old_note:
        note += ' ' + old_note

    args = {'token': api_key['access_token'],
            'Event[Notes]': note}
    resp = requests.put(api_url, data = args)
    logger.info(f"Put new note with return code: {resp} {resp.text}")


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


def process_notes(pending):
    logger.info("**Processing any pending notes")
    while not pending.empty():
        try:
            note, event = pending.get_nowait()
        except queue.Empty:
            logger.info("Breaking due to queue empty")
            break
        except:
            logger.exception("Got exception when trying to get_nowait")

        logger.info(f"Saving note {note} for event {event}")
        save_note(note, event)
    logger.info("Completed processing notes")


if __name__ == "__main__":
    init_logging()
    zones = load_zones(3)
    init()
    last_event = None
    logger.info("Waiting for alarm...")
    force_reload = False
    count = 0
    os.makedirs(config['image_directory'], exist_ok = True)
    pending_notes = queue.SimpleQueue()
    event_closed = True
    while True:
        try:
            count += 1
            if not mem3.is_valid() or force_reload:
                mem3.reload()
                force_reload = False
        except (FileNotFoundError, ValueError, NameError):
            logger.error(f"Error occured while initalizing memory at count: {count}")
            time.sleep(2)  # Wait a couple of seconds, then force a reload and try again.
            if count >= 15:
                # If we aren't getting anywhere with a reload, try a full re-initalization
                logger.warning("Running full memory re-initalization")
                logger.warning("Re-Initalizing")
                init()
                count = 0
            else:
                force_reload = True

            continue

        count = 0
        event_id = mem3.last_event()

        # If we haven't checked before, just use this one
        if last_event is None:
            last_event = event_id

        # Don't try to react to event ID 0
        if event_id and (event_closed == False or event_id != last_event):
            note = None
            if event_id != last_event and event_closed:
                logger.info(f"New Event Detected ID: {event_id}")

            event_data = mem3.get_shared_data()
            if event_data['state'] not in (1, 2, 3):
                if event_closed == True:
                    process_notes(pending_notes)
                    logger.info("Not in alarm. Will keep checking.")
                event_closed = False
                time.sleep(.1)
                continue
            else:
                logger.info("Alarm detected. Analyzing images.")

            t_start = time.time()
            matched, all_objects, match_info, image = process_event(event_id)

            # We have handled this event, so close it
            last_event = event_id
            event_closed = True
            if matched:
                try:
                    requests.get(conf_ini['action']['action_url'])
                except Exception:
                    logger.exception("Unable to call video")
                    pass

                note = save_image(all_objects, match_info, image, event_id)
            logger.info(f"Event processed in {time.time() - t_start}")
            process_notes(pending_notes)
            if note is not None:
                pending_notes.put((note, event_id))

        time.sleep(.1)
