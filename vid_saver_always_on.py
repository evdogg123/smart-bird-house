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
import math

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
        self.before_queue = queue.Queue(maxsize=100) #Magic number, might need to be changed
    
    def __draw_label(self, img, text, pos, bg_color):
        font_face = cv.FONT_HERSHEY_SIMPLEX
        scale = 2
        color = (0, 0, 0)
        thickness = cv.FILLED
        margin = 2

        txt_size = cv.getTextSize(text, font_face, scale, thickness)

        end_x = pos[0] + txt_size[0][0] + margin
        end_y = pos[1] - txt_size[0][1] - margin

        cv.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
        cv.putText(img, text, pos, font_face, scale, color, 1, cv.LINE_AA)

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

    def set_detectedString(self,val):
        self.detectionString = val

    def set_reference_frame(self):
        self.reference_frame = self.oldest_frame.copy()

    def get_reference_frame(self):
        return self.reference_frame

    def reset_reference_frame(self):
        self.reference_frame = []

    def start_stream_processor(self):
        try:
            cap = cv.VideoCapture("http://192.168.1.245:8080/stream/video.mjpeg")
            ret,frame = cap.read()
            while True:


                self.bird_detected = True #Testing only, should use ML model
                ret, frame = cap.read()
                
                if self.before_queue.full():
                    self.oldest_frame = self.before_queue.get()
                    self.before_queue.put(frame)
                else:
                    self.before_queue.put(frame)
                

                
                self.__draw_label(frame, self.detectionString, (50,50), (255,255,255))
                cv.imshow('frame',frame)

                if self.motion_event or write_video: ## Would this actually work??
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
    predictions = []
    while write_video or stream_processor.get_motion_event():
        
        if stream_processor.get_image_arr():
            image = stream_processor.get_latest_image()
            bbox = bb_draw(stream_processor.get_reference_frame(), image)
          
            left, top, right, bottom = bbox
            if left != -1:
                image = image[top:bottom, left:right].copy()
            image = cv.resize(image, (150,150))
            image = np.expand_dims(image,axis=0)
            prediction = bird_classifier.classify_image(image)
            print(prediction)
            predictions.append(prediction)
            
            if predictions[-1] < .15:
                stream_processor.set_detectedString("BIRD")
            else:
                stream_processor.set_detectedString("NOT BIRD OR SQUIRREL")
    
    predictions = np.array(predictions)
   
    average = predictions.mean()
    print(average)
    stream_processor.reset_reference_frame()
        



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
    out = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*'DIVX'),24, (height,width))
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


def bb_draw(reference_img, new_img):
    #reference_img = cv.resize(reference_img, (250,250))
    #new_img = cv.resize(new_img, (250,250))
    '''
    Determines if a new_img contains an object that 
    was not present in reference_img, returns a 150x150 bounding box around the object.
    '''
    reference_img = cv.cvtColor(reference_img, cv.COLOR_BGR2GRAY)
    new_img =  cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)

    reference_img = cv.GaussianBlur(reference_img, (21, 21), 0)
    new_img = cv.GaussianBlur(new_img, (21, 21), 0)
    
    delta_frame = cv.absdiff(reference_img, new_img)
    th_delta = cv.threshold(delta_frame, 20, 255, cv.THRESH_BINARY)[1]
    th_delta = cv.dilate(th_delta, None, iterations=0)
    (cnts, _) = cv.findContours(th_delta.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest_contour = cnts[0]
        for contour in cnts:
            if cv.contourArea(contour) > cv.contourArea(largest_contour):
                largest_contour = contour

        #  and  cv.contourArea(largest_contour) < 7000 In the if staement
        if cv.contourArea(largest_contour) > 5000:
            (x, y, w, h) = cv.boundingRect(largest_contour)
            birdcorner1 = [x,y]
            birdcorner2 = [x+w, y+h]
            birdCenter = [math.floor((birdcorner1[0]+birdcorner2[0])/2), math.ceil((birdcorner1[1]+birdcorner2[1])/2)]
            left,right,top,bottom = birdCenter[0]-100,birdCenter[0]+100,birdCenter[1]-100,birdCenter[1]+100
            height, width = reference_img.shape
    
            if left < 0:
                right -= left
                left = 0
            if top < 0:
                bottom -= top
                top = 0
            if right > width:
                left -= (right - width)
                right = width
            if bottom > height:
                top -= (bottom - height)
                bottom = height
            return (left, top, right, bottom)

    return (-1,-1,-1,-1)


if __name__ == '__main__':
    motion_event_start = False
    motion_event_end = False
    write_video = False
    stream_processor = StreamProcessor()
    bird_classifier = BirdClassifier("/home/bloke/Documents/Birdshit/smart-bird-house/best_model_transfer.hdf5")
    stream_thread = threading.Thread(target=start_stream_process, args=(stream_processor,))
    stream_thread.start()
    classifier_thread = threading.Thread(target=classifier)

@api.route('/birdDetect', methods=['GET'])
def startCam():
    print("WE GOT A REQUEST. MOTION STARTED<<<<<")
    stream_processor.set_motion_event(True)
    stream_processor.set_reference_frame()
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



