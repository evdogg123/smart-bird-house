
import cv2 as cv
import time
import tensorflow as tf
from tensorflow import keras
import scipy.misc
import numpy as np

def load_model(model_dir):
    model = keras.models.load_model(model_dir)
    prob_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    return prob_model
    

probability_model = load_model("my_model")

cap = cv.VideoCapture(-1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    predict_frame = cv.resize(frame, (150,150))
    predict_frame = np.array([predict_frame])
    
    prediction = probability_model.predict(predict_frame, batch_size=1)
    print(prediction[0])

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
