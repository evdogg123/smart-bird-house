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
import sys   
import copy
import math




api = Flask(__name__)
model = None
class StreamProcessor:
    def __init__(self):
        self.image_arr = []
        self.classifier_imgs = []
        self.motion_event = False
        self.prev_motion_event = False
        self.bird_detected = True
        self.detectionString = "BIRD"
        self.before_queue = queue.Queue(maxsize=80) #Magic number, might need to be changed
    
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

    def get_prev_motion_event(self):
        return self.prev_motion_event

    def reset_classifier_imgs(self):
        self.classifier_imgs = []

    def get_classifier_imgs(self):
        return self.classifier_imgs

    def get_latest_image(self):
        return self.classifier_imgs[-1]

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

    def start_stream_processor(self):
        timeout = 0
        prev_motion_event = False
        bbox = (-1,-1,-1,-1)
        print("here")
        try:
            cap = cv.VideoCapture("http://192.168.1.245:8080/stream/video.mjpeg")
            #cap = cv.VideoCapture(-1)
            ret,frame = cap.read()
            oldest_frame = []
            while True:
                
                self.bird_detected = True #Testing only, should use ML model
                ret, frame = cap.read()
                if self.before_queue.full():
                    oldest_frame = self.before_queue.get()
                    self.before_queue.put(frame)

                else:
                    self.before_queue.put(frame)
                if len(oldest_frame)>0:
                    bbox = bb_draw(oldest_frame, frame)
                
                if not prev_motion_event and self.get_motion_event():
                    self.set_image_arr(list(stream_processor.get_before_queue().queue))
    
                if bbox[0] != -1:
                    self.set_motion_event(True)
                    (left, top, right, bottom) = bbox
                    cropped_frame = frame[top:bottom, left:right].copy()
                    cv.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 3)
                    self.classifier_imgs.append(cropped_frame)
                    timeout = 0

                else:
                    timeout += 1
                    if timeout > 50:
                        self.set_motion_event(False)
                        timeout = 0

                self.prev_motion_event = self.get_motion_event()
                    
                self.__draw_label(frame, self.detectionString, (50,50), (255,255,255))

                cv.imshow('frame',frame)

                if self.motion_event:
                    self.image_arr.append(frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            print("caught exception, stream stopped unexpectedly")
            print(e)
            print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
            
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
    while stream_processor.get_motion_event():
        
        if stream_processor.get_classifier_imgs():
            image = stream_processor.get_latest_image()
            print(image)
            #image = cv.resize(image, (150,150))
            image = np.expand_dims(image,axis=0)
            prediction = bird_classifier.classify_image(image)
            print(prediction)
            predictions.append(prediction)
            
            if predictions[-1] < .15:
                stream_processor.set_detectedString("BIRD")
            else:
                stream_processor.set_detectedString("SQUIRREL")
    stream_processor.reset_classifier_imgs()
    predictions = np.array(predictions)
    average = predictions.mean()
    print(average)
        



def write_video_to_disk():
    print("writing video......")
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

def bb_draw(reference_img, new_img):


    reference_img = cv.cvtColor(reference_img, cv.COLOR_BGR2GRAY)
    new_img =  cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)

    reference_img = cv.GaussianBlur(reference_img, (21, 21), 0)
    new_img = cv.GaussianBlur(new_img, (21, 21), 0)
    
    delta_frame = cv.absdiff(reference_img, new_img)
    th_delta = cv.threshold(delta_frame, 20, 255, cv.THRESH_BINARY)[1]
    th_delta = cv.dilate(th_delta, None, iterations=0)
    (cnts, _) = cv.findContours(th_delta.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)



    largest_contour = cnts[0]
    for contour in cnts:
        if cv.contourArea(contour) > cv.contourArea(largest_contour):
            largest_contour = contour

    
    if cv.contourArea(largest_contour) > 1500:
        (x, y, w, h) = cv.boundingRect(largest_contour)
        birdcorner1 = [x,y]
        birdcorner2 = [x+w, y+h]
        birdCenter = [math.floor((birdcorner1[0]+birdcorner2[0])/2), math.ceil((birdcorner1[1]+birdcorner2[1])/2)]
        left = int(birdCenter[0]-75)
        right = int(birdCenter[0]+75)
        top = int(birdCenter[1]-75)
        bottom = int(birdCenter[1]+75)
        height, width = reference_img.shape
        if left < 0:
            diff = -1 * left
            right += diff
            left = 0
        if top < 0:
            diff = -1 * top
            bottom += diff
            top = 0
        if right > width:
            diff = right - width
            left -= diff
            right = width
        if bottom > height:
            diff = bottom - height
            top -= diff
            bottom = height
        return (left, top, right, bottom)
    else:
        return (-1,-1,-1,-1)


if __name__ == '__main__':

    write_video = False
    motion_event_retriggered = False

    stream_processor = StreamProcessor()
    bird_classifier = BirdClassifier("/home/bloke/Documents/Birdshit/smart-bird-house/best_model_transfer.hdf5")
    stream_thread = threading.Thread(target=start_stream_process, args=(stream_processor,))
    stream_thread.start()
    classifier_thread = threading.Thread(target=classifier)

    while True:
        if stream_processor.get_motion_event() and  not stream_processor.get_prev_motion_event():
            print("MOTION EVENT START>>>>>>>>>>")
            if not classifier_thread.is_alive():
                classifier_thread =  threading.Thread(target=classifier)
                classifier_thread.start()

        elif not stream_processor.get_motion_event() and stream_processor.get_prev_motion_event():
            print("MOTION EVENT END<<<<<<<<<<<")
            if stream_processor.get_bird_detected() and not write_video:
                save_thread = threading.Thread(target=write_video_to_disk)
                save_thread.start()
      




 
