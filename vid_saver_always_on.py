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
api = Flask(__name__)

class StreamProcessor:

    def __init__(self):
        self.image_arr = []
        self.motion_event = False
        self.bird_detected = True

    def get_image_arr(self):
        return self.image_arr

    def reset_img_arr(self):
        self.image_arr = []

    def set_motion_event(self,val):
        self.motion_event = val

    def get_motion_event(self):
        return self.motion_event
    
    def get_bird_detected(self):
        return self.bird_detected

    def set_bird_detected(self,val):
        self.bird_detected = val

    def start_stream_processor(self):
        cap = cv.VideoCapture("http://192.168.1.245:8000/stream.mjpg")
        try:
            ret,frame = cap.read()
            while True:

                self.bird_detected = True #Testing only, should use ML model
                ret, frame = cap.read()
                
                if not ret:
                    break
                cv.imshow('frame',frame)

                if self.motion_event:
                    self.image_arr.append(frame)

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

def start_stream_process(stream_processor):
    stream_processor.start_stream_processor()


stream_processor = StreamProcessor()
stream_thread = threading.Thread(target=start_stream_process, args=(stream_processor,))


@api.route('/birdDetect', methods=['GET'])
def startCam():
    stream_processor.set_motion_event(True)
    print("WE GOT A REQUEST. MOTION STARTED<<<<<")
    if not stream_thread.is_alive():
        stream_thread.start()
    return json.dumps({"sucsess": True})
    
   
@api.route('/birdEnd', methods=['GET'])
def endCam():
    print("WE GOT A REQUEST. MOTION ENDED>>>>>")
    stream_processor.set_motion_event(False)
    if stream_processor.get_bird_detected():
        bird_video = deepcopy(stream_processor.get_image_arr())
        save_thread = threading.Thread(target=write_video,args=(bird_video,))
        save_thread.start()
        stream_processor.set_bird_detected(False)
    stream_processor.reset_img_arr()
    return json.dumps({"sucsess": True})


if __name__ == '__main__':
    api.run(host= '0.0.0.0')

