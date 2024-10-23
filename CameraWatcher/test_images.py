import configparser
import logging
import os
import re
import time
import io

import cv2
import numpy
import requests


from PIL import Image
import PIL
print("PIL version:", PIL.__version__)

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
        # logger.warning("Error when retrieving file!")
        raise requests.exceptions.HTTPError("Bad Content")

    return res.content

img_data = download_zm_image()

numpy_img = numpy.frombuffer(img_data, numpy.uint8)
cv_img = cv2.imdecode(numpy_img, cv2.IMREAD_COLOR)

pil_image = Image.open(io.BytesIO(img_data))

print("OpenCV Image Size:", cv_img.shape)
print("pil image size:", pil_image.size)

scale_percent = 15.625
dest_dim = (300, 169)

url = 'http://localhost/zm/cgi-bin/nph-zms?monitor=3&scale=100&maxfps=30&buffer=1000&user=israel&pass=shanima81'
cap = cv2.VideoCapture(url)

def cv_resize():
    #ret, cv_img = cap.read()
    
    img_data = download_zm_image()
    numpy_img = numpy.frombuffer(img_data, numpy.uint8)
    cv_img = cv2.imdecode(numpy_img, cv2.IMREAD_COLOR)
    
    new_img = cv2.resize(cv_img, dest_dim, interpolation = cv2.INTER_AREA)
    return new_img

def pil_resize():
    ret, cv_img = cap.read()
    # img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_img[:, :, ::-1])
    
#    img_data = download_zm_image()
#    pil_image = Image.open(io.BytesIO(img_data))
    
    new_img = pil_image.resize(dest_dim, reducing_gap = 2.0)
    return new_img

print("My PID is:", os.getpid())
start = time.time()
end = start + 30
count = 0
while time.time() < end:
    t1 = time.time()
    img = pil_resize()
    count += 1
    if time.time() - t1 < .1:
        time.sleep(.1 - (time.time() - t1))    
 
cap.release()

#cv2.imwrite('/tmp/out_images/test_cv.png', img)
#img.save("/tmp/out_images/test_pil.png")

runtime = time.time() - start
print(f"Resized {count} times in {runtime} seconds for {count / runtime} fps")

print("DONE!")
