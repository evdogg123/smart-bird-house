import cv2 as cv
import time
import tensorflow as tf
from tensorflow import keras
import scipy.misc
import numpy as np
import threading
from datetime import datetime
import requests
from flask import Flask, json
from copy import deepcopy


from flask import Flask, json
stream = False
refresh_request = True #testing only
threshold = 0 #testing only
api = Flask(__name__)
image_arr = []
motion_event = False
bird_detected = True


def stream_processor():
    cap = cv.VideoCapture("http://192.168.1.245:8000/stream.mjpg")
    global bird_detected
    
    stream = True
    try:
        ret,frame = cap.read()
        while stream:

            bird_detected = True #Testing only, should use ML model
            ret, frame = cap.read()
            print(ret)

            if not ret:
                break
            cv.imshow('frame',frame)

            if motion_event:
                image_arr.append(frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break


    except Exception as e:
        print("caught exception, stream stopped unexpectedly")
        print(e)
    finally:    
        cap.release()
        cv.destroyAllWindows()


def write_video(image_array):
    capture_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    vid_name = "vid" + capture_time + ".avi"
    height, width, _ = image_array[0].shape
    out = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*'DIVX'),30, (height,width))
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()         


stream_thread = threading.Thread(target=stream_processor)

@api.route('/birdDetect', methods=['GET'])
def startCam():
    global motion_event
    motion_event = True
    print("WE GOT A REQUEST. MOTION STARTED<<<<<")
    if not stream_thread.is_alive():
        stream_thread.start()
    return json.dumps({"sucsess": True})
    
   
    

@api.route('/birdEnd', methods=['GET'])
def endCam():
    global motion_event
    motion_event = False
    print("WE GOT A REQUEST. MOTION ENDED>>>>>")
    global bird_detected
    global image_arr
    if bird_detected:
        bird_video = deepcopy(image_arr)
        write_video(bird_video)
        bird_detected = False
    image_arr = []
    return json.dumps({"sucsess": True})


if __name__ == '__main__':
    
  
    api.run(host= '0.0.0.0')

