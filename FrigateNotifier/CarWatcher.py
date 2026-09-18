#!/usr/bin/env python3
"""
High-rate car position tracker via Frigate API polling + homography.
Publishes to MQTT for Home Assistant.
"""

import json
import time
import requests
import numpy as np
import cv2
from collections import deque
import paho.mqtt.client as mqtt

# ============== CONFIG ==============
FRIGATE_URL       = "http://frigate:5000"          # or your IP
CAMERA_NAME       = "garage"                       # exact camera name
POLL_INTERVAL     = 0.2                            # 5 Hz

# Homography (replace after calibration)
H = np.eye(3, dtype=np.float64)

# MQTT output
MQTT_HOST         = "your-mqtt-broker"
MQTT_PORT         = 1883
MQTT_USER         = None
MQTT_PASS         = None
OUT_PREFIX        = "garage/car"

HISTORY_SEC       = 2.5
MOVE_THRESHOLD_M  = 0.18                           # tune this
# ====================================

session = requests.Session()
history = deque()          # (timestamp, x, y)

def get_current_car_box():
    """Return the most recent in-progress car box or None."""
    try:
        r = session.get(
            f"{FRIGATE_URL}/api/events",
            params={
                "camera": CAMERA_NAME,
                "label": "car",
                "in_progress": 1,
                "limit": 5,
            },
            timeout=1.0,
        )
        r.raise_for_status()
        events = r.json()
        if not events:
            return None

        # Take the newest one
        event = max(events, key=lambda e: e.get("start_time", 0))
        data = event.get("data", {})
        box = data.get("box")          # usually [x1, y1, x2, y2] or normalized
        if not box or len(box) < 4:
            return None

        # Handle both pixel and normalized formats
        if max(box) <= 1.5:            # normalized
            # You need the detect resolution for this camera
            # For now assume you know width/height or fetch from config
            # Example: box = [x, y, w, h] normalized → convert
            pass

        return box, event.get("id"), data.get("score", 0)
    except Exception as e:
        print(f"API error: {e}")
        return None

def front_point(box, method="bottom_right"):
    """
    box = [x1, y1, x2, y2]  (left, top, right, bottom)
    Returns (u, v) image coordinates of the front of the car.
    """
    x1, y1, x2, y2 = box

    if method == "bottom_right":
        return x2, y2                    # classic ground-plane point at front

    elif method == "right_center":
        return x2, (y1 + y2) / 2.0       # mid-height on the front edge

    elif method == "right_low":
        # 75 % of the way down the right edge – good compromise
        return x2, y1 + 0.75 * (y2 - y1)

    else:
        raise ValueError(f"Unknown method: {method}")

def image_to_world(u, v):
    p = np.array([[[u, v]]], dtype=np.float32)
    world = cv2.perspectiveTransform(p, H)[0, 0]
    return float(world[0]), float(world[1])

def main():
    client = mqtt.Client()
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()

    print("Starting high-rate car tracker...")

    while True:
        start = time.time()
        result = get_current_car_box()

        if result is None:
            # No car currently tracked
            client.publish(f"{OUT_PREFIX}/present", "OFF", retain=True)
            client.publish(f"{OUT_PREFIX}/moving", "OFF", retain=True)
            history.clear()
            time.sleep(POLL_INTERVAL)
            continue

        box, event_id, score = result
        u, v = front_point(box)
        x, y = image_to_world(u, v)

        now = time.time()
        history.append((now, x, y))

        # Keep only recent history
        while history and now - history[0][0] > HISTORY_SEC:
            history.popleft()

        # Speed & moving flag (your definition, not Frigate’s)
        speed = 0.0
        moving = False
        if len(history) >= 2:
            t0, x0, y0 = history[0]
            t1, x1, y1 = history[-1]
            dt = t1 - t0
            if dt > 0.15:
                dist = np.hypot(x1 - x0, y1 - y0)
                speed = dist / dt
                moving = dist > MOVE_THRESHOLD_M

        payload = {
            "x": round(x, 3),
            "y": round(y, 3),
            "distance": round(x, 3),          # adjust axis as needed
            "speed_mps": round(speed, 2),
            "moving": moving,
            "score": score,
            "event_id": event_id,
            "ts": now,
        }

        client.publish(f"{OUT_PREFIX}/state", json.dumps(payload), retain=True)
        client.publish(f"{OUT_PREFIX}/present", "ON", retain=True)
        client.publish(f"{OUT_PREFIX}/distance", payload["distance"], retain=True)
        client.publish(f"{OUT_PREFIX}/moving", "ON" if moving else "OFF", retain=True)
        client.publish(f"{OUT_PREFIX}/speed", payload["speed_mps"], retain=True)

        # Adaptive sleep to hit target rate
        elapsed = time.time() - start
        time.sleep(max(0.01, POLL_INTERVAL - elapsed))

if __name__ == "__main__":
    main()