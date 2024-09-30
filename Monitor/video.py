import logging
import math
import os
import threading
import time

from functools import lru_cache

import cv2

from shapely.geometry import Polygon, box

from . import config

logger = logging.getLogger("cameramon")

t0 = None
def init_video_stream():
    global t0
    # Set FFMPEG options to minimize delay and eliminate buffering 
    # so we always get the most recent frame (hopefully)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|probesize;65536|analyzeduration;100000|framedrop;1|flags;low_delay|fflags;nobuffer"
    url = config['frame']['rtsp_source']

    cap = cv2.VideoCapture(url)
    backend_name = cap.getBackendName()
    print(f"Using backend: {backend_name}")
    if backend_name == "GSTREAMER":
        # Use a GStreamer pipeline to control buffering
        pipeline = f"rtspsrc location={url} latency=0 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true"
        cap.release()  # Release the previous VideoCapture
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        print("Configured GStreamer pipeline.")    
    
    # Another attempt to eliminate buffering
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