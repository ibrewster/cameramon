import os
from collections import defaultdict

import cv2
import time
import imutils

import zmdetect_fast2
from zmdetect_fast2 import process_event, load_zones, init_logging, init

if __name__ == "__main__":
    global zones
    global logger
    os.makedirs('/tmp/zmdetect', exist_ok = True)

    init_logging()
    init()

    logger = zmdetect_fast2.logger
    print("Logger is:", logger, "<<<>>>")

    zones = load_zones(3)
    zmdetect_fast2.zones = zones

    match_other_image = False
    for idx in range(5):
        # t1 = time.time()
        matched, all_objects, match_info, image = process_event(idx)
        if matched:
            print("**************************")
            print("Found match in loop", idx)
            print("**************************")
            if idx != 0:
                match_other_image = True
        # t2 = time.time()
        time.sleep(.25)

    if match_other_image:
        print("!!!!!!!!!!!!Got match outside first image!!!!!!!!!")
#     matched, all_objects, match_info, image = process_event(2)
#     print("Found match second time:", matched)
#     t3 = time.time()
#
#     matched, all_objects, match_info, image = process_event(3)
#     print("Found match third time:", matched)
#     t4 = time.time()
#     print("Ran first detect in", t2 - t1)
#     print("Ran second detect in", t3 - t2)
#     print("Ran third detect in", t4 - t3)
