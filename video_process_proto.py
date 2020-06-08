import cv2 as cv
import time
import tensorflow as tf
from tensorflow import keras
import scipy.misc
import numpy as np
import threading
from datetime import datetime
import asyncio
import requests


def load_model(model_dir):
    model = keras.models.load_model(model_dir)
    prob_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    return prob_model

#Webcam
#cap = cv.VideoCapture(0)
#Pi Stream
cap = cv.VideoCapture("http://192.168.1.245:8000/stream.mjpg")
image_arr = []

bird_detected = True #TESTING ONLY CHANGE THIS
refresh_request = True #TESTING ONLY CHANGE THIS
threshold = 0 #TESTING ONLY CHANGE THIS
video_counter = 0 #Should use datetime as unique identifier for saved videos
STREAM_IMG_HEIGHT = 250 #TESTING ONLY CHANGE THIS
STREAM_IMG_WIDTH = 250 #TESTING ONLY CHANGE THIS


def stream_processor():
    try:
        while True:
            ret, frame = cap.read()
            image_arr.append(frame)
            cv.imshow('frame',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print("caught exception, motion event ended")
    finally:
        if bird_detected:
            requests.get(url="http://192.168.1.245:8000/open")
            #save array of images as video
            capture_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
            vid_name = "saved_videos/vid" + capture_time + ".avi"
            out = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*'DIVX'),30, (STREAM_IMG_WIDTH,STREAM_IMG_HEIGHT))
            for i in range(len(image_arr)):
                out.write(image_arr[i])
             
            out.release()


    cap.release()
    cv.destroyAllWindows()



def main():
    while True: #Shouldn't be a spin wait, should just wake up threads
        if refresh_request:
            stream_processor()
        break
           
           
main()
    

   
    




