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
import queue
from flask import Flask, json

write_video = False
motion_event_retriggered = False

api = Flask(__name__)
model = None
class StreamProcessor:
    def __init__(self):
        self.image_arr = []
        self.motion_event = False
        self.bird_detected = True
        self.before_queue = queue.Queue(maxsize=70) #Magic number, might need to be changed

    def get_before_queue(self):
        return self.before_queue

    def get_latest_image(self):
        return self.image_arr[-1]

    def get_image_arr(self):
        return self.image_arr

    def set_image_arr(self,images):
        self.image_arr = images

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
        
        try:
            cap = cv.VideoCapture("http://192.168.1.245:8080/stream/video.mjpeg")
            ret,frame = cap.read()
            while True:


                self.bird_detected = True #Testing only, should use ML model
                ret, frame = cap.read()
                
                if self.before_queue.full():
                    self.before_queue.get()
                    self.before_queue.put(frame)
                else:
                    self.before_queue.put(frame)
                

                if not ret:
                    break
                cv.imshow('frame',frame)

                if self.motion_event or write_video: ## Would this actually work??
                    self.image_arr.append(frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print("caught exception, stream stopped unexpectedly")
            print(e)
        finally:    
            cap.release()
            cv.destroyAllWindows()

class BirdClassifier:
    def __init__(self, path_to_model):
        self.model = self.load_model(path_to_model)
    

    def load_model(self, model_dir):
        model = keras.models.load_model(model_dir)
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        model.summary()
        return model

    
    def classify_image(self, image):
        return self.model.predict(image, batch_size = 1)
    

def classifier():
    while write_video or stream_processor.get_motion_event():
        if stream_processor.get_image_arr():
            image = stream_processor.get_latest_image()
            image = cv.resize(image, (150,150))
            image = np.expand_dims(image,axis=0)
            print(bird_classifier.classify_image(image))
        



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
    out = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*'MPEG'),24, (height,width))
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()

    stream_processor.reset_img_arr()
    print("finished writing video......")
    write_video = False
      

def start_stream_process(stream_processor):
    stream_processor.start_stream_processor()


stream_processor = StreamProcessor()
bird_classifier = BirdClassifier("/home/bloke/Documents/Birdshit/smart-bird-house/best_model_transfer.hdf5")
stream_thread = threading.Thread(target=start_stream_process, args=(stream_processor,))
classifier_thread = threading.Thread(target=classifier)


@api.route('/birdDetect', methods=['GET'])
def startCam():
    print("WE GOT A REQUEST. MOTION STARTED<<<<<")
    stream_processor.set_motion_event(True)
    classifier_thread = threading.Thread(target=classifier)
    classifier_thread.start()
    stream_processor.set_image_arr(list(stream_processor.get_before_queue().queue))
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
    if stream_processor.get_bird_detected() and not write_video:
        print("starting video save...")
        save_thread = threading.Thread(target=write_video_to_disk)
        save_thread.start()
        stream_processor.set_bird_detected(False)
    
    return json.dumps({"success": True})


if __name__ == '__main__':
    api.run(host= '0.0.0.0')

