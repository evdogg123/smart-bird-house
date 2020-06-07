import cv2 as cv2
import time
import numpy as np

cap = cv2.VideoCapture("http://192.168.1.245:8000/stream.mjpg")

while True:
    result, frame = cap.read()
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break