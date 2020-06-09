import cv2 as cv
import time
import tensorflow as tf
from tensorflow import keras
import scipy.misc
import numpy as np
import threading
from datetime import datetime
import requests

cap = cv.VideoCapture(-1)
#cap = cv.VideoCapture("http://192.168.1.245:8000/stream.mjpg")

refresh_request = True #testing only
threshold = 0 #testing only



def load_model(model_dir):
    model = keras.models.load_model(model_dir)
    prob_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    return prob_model
    
probability_model = load_model("ML/model2_save")

def stream_processor():
    STREAM_HEIGHT = 0
    STREAM_WIDTH = 0
    image_arr = []
    bird_detected = True #Should be False, True for testing
    try:
        ret,frame = cap.read()
        while True:
            ret, frame = cap.read()
            cv.imshow('frame',frame)
            image_arr.append(frame)
            predict_frame = cv.resize(frame, (150,150))
            predict_frame = np.array([predict_frame])
            prediction = probability_model.predict(predict_frame, batch_size=1)
            print(prediction[0])
            if prediction[0] > threshold:
                bird_detected = True

            if cv.waitKey(1) & 0xFF == ord('q'):
                break


    except Exception as e:
        print("caught exception, motion event ended")
    finally:
        if bird_detected:
            #Tell birdhouse to open up birdfeeder
            #requests.get(url="http://192.168.1.245:8000/open")
            #save array of images as video
            write_video(image_arr)
    

    cap.release()
    cv.destroyAllWindows()

def write_video(image_array):
    capture_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    vid_name = "saved_videos/vid" + capture_time + ".avi"
    height, width, _ = image_array[0].shape
    out = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*'DIVX'),30, (height,width))
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()         


def main():
    while True: #Shouldn't be a spin wait, should just wake up threads
        if refresh_request:
            stream_processor()
        break #testing
           
main()
    

   