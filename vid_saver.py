import cv2 as cv
import time

import scipy.misc
import numpy as np
import threading
from datetime import datetime
import requests
from flask import Flask, json
from copy import deepcopy
import queue
from flask import Flask, json
import math
import redis

write_video = False
motion_event_retriggered = False


api = Flask(__name__)
model = None


class StreamProcessor:
    def __init__(self):
        self.oldest_frame = []
        self.reference_frame = []
        self.image_arr = []
        self.motion_event = False
        self.bird_detected = True
        self.detectionString = "BIRD"
        self.r = redis.Redis(host='localhost', port=6379, db=0)
        self.pipe = self.r.pubsub()
        self.pipe.subscribe('birdEvents',)
        # Magic number, might need to be changed
        self.before_queue = queue.Queue(maxsize=10)

    def get_before_queue(self):
        return self.before_queue

    def get_latest_image(self):
        return self.image_arr[-1]

    def get_image_arr(self):
        return self.image_arr

    def set_image_arr(self, images):
        self.image_arr = images

    def reset_img_arr(self):
        self.image_arr = []

    def set_motion_event(self, val):
        self.motion_event = val

    def get_motion_event(self):
        return self.motion_event

    def start_stream_processor(self):
        try:
            cap = cv.VideoCapture(
                "http://192.168.1.245:8080/stream/video.mjpeg")
            ret, frame = cap.read()
            while True:

                self.bird_detected = True  # Testing only, should use ML model
                ret, frame = cap.read()

                cv.imshow('frame', frame)

                if self.motion_event or write_video:  # Would this actually work??
                    self.image_arr.append(frame)

                if not ret:
                    break
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print("caught exception, stream stopped unexpectedly")
            print(e)
        finally:
            cap.release()
            cv.destroyAllWindows()


def write_video_to_disk():
    global write_video
    global motion_event_retriggered
    write_video = True
    motion_event_retriggered = True

    while motion_event_retriggered:

        motion_event_retriggered = False
        print("waiting for retrigger event......")
        time.sleep(3.5)
    print("finished recording, writing video......")
    motion_event_retriggered = False

    image_array = deepcopy(stream_processor.get_image_arr())
    capture_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    vid_name = "bird_recordings/vid" + capture_time + ".avi"

    height, width, _ = image_array[0].shape
    out = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(
        *'DIVX'), 24, (height, width))
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()

    stream_processor.reset_img_arr()
    print("finished writing video......")
    dat = {"Time": capture_time}
    stream_processor.pipe.publish("birdEvent", str(dat))
    write_video = False

    """

    thumbnail_name = "static/" + bird_classifier.get_classification() + \
        capture_time + ".jpg"
    
    cv.imwrite(thumbnail_name, bird_classifier.get_thumbnail())
    bird_classifier.set_thumbnail([])
    bird_classifier.set_classification("none")
    """


def start_stream_process(stream_processor):
    stream_processor.start_stream_processor()


stream_processor = StreamProcessor()
stream_thread = threading.Thread(
    target=start_stream_process, args=(stream_processor,))


if __name__ == '__main__':
    motion_event_start = False
    motion_event_end = False
    write_video = False
    stream_processor = StreamProcessor()

    stream_thread = threading.Thread(
        target=start_stream_process, args=(stream_processor,))
    stream_thread.start()


@api.route('/birdDetect', methods=['GET'])
def startCam():
    print("WE GOT A REQUEST. MOTION STARTED<<<<<")
    stream_processor.set_motion_event(True)

    stream_processor.set_image_arr(
        list(stream_processor.get_before_queue().queue))

    global motion_event_retriggered
    if write_video:
        print("Retriggered...")
        motion_event_retriggered = True

    if not stream_thread.is_alive():
        stream_thread.start()
    return json.dumps({"success": True})


@api.route('/birdEnd', methods=['GET'])
def endCam():
    print("WE GOT A REQUEST. MOTION ENDED>>>>>")
    stream_processor.set_motion_event(False)
    if not write_video:
        print("starting video save...")
        save_thread = threading.Thread(target=write_video_to_disk)
        save_thread.start()

    return json.dumps({"success": True})


if __name__ == '__main__':
    api.run(host='0.0.0.0')
