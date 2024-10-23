import os
import glob
import json

from datetime import datetime

import flask
import numpy

from . import app, CONFIG


def event_bin_factory():
    return {'day': 0,
            'week': 0,
            'month': 0,
            'all': 0, }


@app.route('/')
def list_monitors():
    now = datetime.today()
    image_path = CONFIG['action']['image_directory']
    cameras = os.listdir(image_path)
    monitors = {}
    for camera in cameras:
        monitors[camera] = event_bin_factory()
        event_path = os.path.join(image_path, camera)
        monitors[camera]['event_path'] = event_path
        events = os.listdir(event_path)
        for event_dir in events:
            try:
                event_date = datetime.strptime(event_dir, "%Y-%m-%d")
            except ValueError:
                continue # Not a valid event dir
            
            events = glob.glob(os.path.join(event_path, event_dir, "*_objdetect.jpg"))
            event_count = len(events)
            age = (now - event_date).total_seconds() / 60 / 60  # hours
            if age < 24:
                monitors[camera]['day'] += event_count
            if age < 24 * 7:
                monitors[camera]['week'] += event_count
            if age < 24 * 30:
                monitors[camera]['month'] += event_count
            monitors[camera]['all'] += event_count
    return flask.render_template('index.html', monitors = monitors)


@app.route('/getEvents')
def list_events():
    now = datetime.today()
    image_path = CONFIG['action']['image_directory']
    monitor = flask.request.args['monitor']
    age_type = flask.request.args['type']
    max_age = None
    if age_type == 'day':
        max_age = 24
    elif age_type == 'week':
        max_age = 24 * 7
    elif age_type == 'month':
        max_age = 24 * 30

    event_path = os.path.join(image_path, monitor)
    events = []
    for date in sorted(os.listdir(event_path), reverse = True):
        event_date = datetime.strptime(date, "%Y-%m-%d")
        event_age = (now - event_date).total_seconds() / 60 / 60
        if max_age is None or event_age < max_age:
            for event in sorted(glob.glob(os.path.join(event_path, date, "*.jpg")), reverse = True):
                img_name = os.path.basename(event)
                event_name = img_name[:8]
                event_info_file = event_name + "_objects.json"
                with open(os.path.join(event_path, date, event_info_file), 'r') as evinfo:
                    event_info = json.load(evinfo)
                labels = event_info['labels']
                confs = (numpy.round(event_info['confidences'], 2) * 100).astype(int)
                items = [f"{label.capitalize()} ({conf}%)" for label, conf in zip(labels, confs)]

                img_time = datetime.strptime(event_name, "%H-%M-%S").strftime("%H:%M:%S")
                events.append((event_date.strftime("%-m/%-d/%y"), img_time,
                               ", ".join(items)))

    return flask.render_template('events.html', monitor = monitor, events = events)


@app.route('/getImage')
def get_image():
    image_path = CONFIG['action']['image_directory']
    monitor = flask.request.args['monitor']
    date = flask.request.args['date']
    time = flask.request.args['time']
    date_time = datetime.strptime(date + "T" + time,
                                  "%m/%d/%yT%H:%M:%S")
    date = date_time.strftime('%Y-%m-%d')
    time = date_time.strftime('%H-%M-%S')
    image_path = os.path.join(image_path, monitor, date, time + "_objdetect.jpg")
    return flask.send_file(image_path)
