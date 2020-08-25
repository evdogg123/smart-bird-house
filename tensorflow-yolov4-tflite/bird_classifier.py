from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import numpy as np
import cv2 as cv
import json
import os
import tensorflow as tf
from tensorflow import keras
import redis

import operator
from PIL import Image
from tensorflow.python.saved_model import tag_constants
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score',  0.25, 'score threshold')


BIRD_CLASS = 14


class YoloClassifier:

    def __init__(self, model, image_size):
        self.model = model
        self.image_size = image_size
        self.images = []
        self.bird_detected = False
        self.bird_images = []
        self.confidence_arr = []
        self.thumbnail_center = []
        self.thumbnail = []
        self.vid_name = ""

    def clear_data(self):
        self.images = []
        self.bird_detected = False
        self.bird_images = []
        self.confidence_arr = []
        self.thumbnail_center = []
        self.thumbnail = []
        self.vid_name = ""

    def get_crops(self):
        return self.bird_images

    def load_video(self, video_file):
        # Loads video frames into image array
        video_name = video_file.replace(".avi", "")
        self.vid_name = video_name
        cap = cv.VideoCapture("../bird_recordings/" + video_file)
        while True:
            ret, frame = cap.read()
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            if not ret:
                break
            cv.imshow("frame", frame)
            self.images.append(frame)
        self.images = np.asarray(self.images).astype(np.float32)

    def classify(self):
        """
        -Determines if birds are present in the video using YOLOv4 architecture
        -Crops image based on bounding box that contains the bird
        """
        infer = self.model.signatures['serving_default']
        for i, original_image in enumerate(self.images):
            image = original_image.copy()
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (self.image_size, self.image_size))
            image = image / 255.

            image = [image]
            image = np.asarray(image).astype(np.float32)
            batch_data = tf.constant(image)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=10,
                max_total_size=10,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            height, width, _ = original_image.shape

            print(scores)
            classes = classes[0]
            print(classes)

            bbox = boxes[0][0].numpy()
            bbox[0] = int(bbox[0] * height)
            bbox[2] = int(bbox[2] * height)
            bbox[1] = int(bbox[1] * width)
            bbox[3] = int(bbox[3] * width)

            if BIRD_CLASS in classes:
                idx = np.where(classes == BIRD_CLASS)
                bbox = bbox.astype(np.int)
                x = int((bbox[1] + bbox[3]) / 2)
                y = int((bbox[0] + bbox[2]) / 2)
                self.thumbnail_center.append((x, y))
                cropped_img = original_image[bbox[0]:bbox[2], bbox[1]: bbox[3]]
                self.bird_images.append(cropped_img)
                self.confidence_arr.append(scores[idx[0][0]][0])

        self.generate_thumbnail(size=150)

    def generate_thumbnail(self, size=150):
        # Generates a thumbnail image of the bird of size "size"
        idx = np.argmax(self.confidence_arr)
        print(self.confidence_arr[idx])
        print(idx)
        best_image = self.images[idx]
        x, y = self.thumbnail_center[idx]
        offset = int(size / 2)
        self.thumbnail = best_image[x -
                                    offset: x + offset, y - offset: y + offset]
        print("../static/thumbnails/thumb_" +
              self.vid_name + ".jpg")
        cv.imwrite("../static/thumbnails/thumb_" +
                   self.vid_name + ".jpg", self.thumbnail)


class SpeciesClassifier:
    def __init__(self, model_dir):
        self.model = self.load_model(model_dir)
        self.pred_dict = {}
        self.labels = self.load_labels()

    def load_model(self, model_dir):
        model = keras.models.load_model(model_dir)
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        return model

    def reset_pred_dict(self):
        self.pred_dict = {}

    def classify(self, images, confidence_arr):
        for i, image in enumerate(images):
            image = cv.resize(image, (299, 299))
            image = image / 255.
            image = np.expand_dims(image, axis=0)
            prediction = self.model.predict(image, batch_size=1)
            print(prediction)
            self.update_pred(prediction, confidence_arr[i])
            print("CONFIDENCE")
            print(confidence_arr[i])

        print("PREDICTION DICTIONARY")
        print(self.pred_dict)
        pred = max(self.pred_dict.items(), key=operator.itemgetter(1))[0]

        top_three = sorted(self.pred_dict.items(),
                           key=lambda x: x[1], reverse=True)

        preds = [bird[0] for bird in top_three[0:3]]
        print(preds)
        return preds

    def load_labels(self):
        labels = {0: 'AMERICAN CROW', 1: 'AMERICAN GOLDFINCH (BREEDING MALE)', 2: 'AMERICAN ROBIN (ADULT)', 3: 'AMERICAN ROBIN (JUVENILE)', 4: 'AMERICAN TREE SPARROW', 5: "ANNA'S HUMMINGBIRD (ADULT MALE)", 6: "ANNA'S HUMMINGBIRD (FEMALE, IMMATURE)", 7: 'BALTIMORE ORIOLE (ADULT MALE)', 8: 'BAND-TAILED PIGEON', 9: "BEWICK'S WREN", 10: 'BLACK-BILLED MAGPIE', 11: 'BLACK-CAPPED CHICKADEE', 12: 'BLACK-CHINNED HUMMINGBIRD (ADULT MALE)', 13: 'BLACK-CHINNED HUMMINGBIRD (FEMALE, IMMATURE)', 14: 'BLACK-CRESTED TITMOUSE', 15: 'BLUE JAY', 16: 'BOHEMIAN WAXWING', 17: 'BOREAL CHICKADEE', 18: "BREWER'S BLACKBIRD (MALE)", 19: 'BROWN CREEPER', 20: 'BROWN THRASHER', 21: 'BROWN-HEADED COWBIRD (MALE)', 22: 'BROWN-HEADED NUTHATCH', 23: 'BUSHTIT', 24: 'CACTUS WREN', 25: 'CALIFORNIA QUAIL (MALE)', 26: 'CALIFORNIA TOWHEE', 27: 'CANYON TOWHEE', 28: 'CAROLINA CHICKADEE', 29: 'CAROLINA WREN', 30: "CASSIN'S FINCH (ADULT MALE)", 31: 'CEDAR WAXWING', 32: 'CHESTNUT-BACKED CHICKADEE', 33: 'CHIPPING SPARROW (BREEDING)', 34: "CLARK'S NUTCRACKER", 35: 'COMMON GRACKLE', 36: 'COMMON GROUND-DOVE', 37: 'COMMON RAVEN', 38: 'COMMON REDPOLL', 39: 'CURVE-BILLED THRASHER', 40: 'DARK-EYED JUNCO (OREGON)', 41: 'DARK-EYED JUNCO (PINK-SIDED)', 42: 'DARK-EYED JUNCO (SLATE-COLORED)', 43: 'DARK-EYED JUNCO (WHITE-WINGED)', 44: 'DOWNY WOODPECKER', 45: 'EASTERN BLUEBIRD', 46: 'EASTERN MEADOWLARK', 47: 'EASTERN TOWHEE', 48: 'EURASIAN COLLARED-DOVE', 49: 'EUROPEAN STARLING (BREEDING ADULT)', 50: 'EUROPEAN STARLING (JUVENILE)', 51: 'EUROPEAN STARLING (NONBREEDING ADULT)', 52: 'EVENING GROSBEAK (ADULT MALE)', 53: 'FIELD SPARROW', 54: 'FOX SPARROW (RED)', 55: 'FOX SPARROW (SOOTY)',
                  56: "GAMBEL'S QUAIL (MALE)", 57: 'GOLDEN-CROWNED SPARROW (ADULT)', 58: 'GOLDEN-CROWNED SPARROW (IMMATURE)', 59: 'GRAY JAY', 60: 'GREAT-TAILED GRACKLE', 61: 'HAIRY WOODPECKER', 62: "HARRIS'S SPARROW (ADULT)", 63: "HARRIS'S SPARROW (IMMATURE)", 64: 'HERMIT THRUSH', 65: 'HOARY REDPOLL', 66: 'HOUSE FINCH (ADULT MALE)', 67: 'HOUSE SPARROW (MALE)', 68: 'INCA DOVE', 69: 'JUNIPER TITMOUSE', 70: 'LESSER GOLDFINCH (ADULT MALE)', 71: 'MOUNTAIN CHICKADEE', 72: 'MOURNING DOVE', 73: 'NORTHERN CARDINAL (ADULT MALE)', 74: 'NORTHERN FLICKER (RED-SHAFTED)', 75: 'NORTHERN FLICKER (YELLOW-SHAFTED)', 76: 'NORTHERN MOCKINGBIRD', 77: 'OAK TITMOUSE', 78: 'PILEATED WOODPECKER', 79: 'PINE GROSBEAK (ADULT MALE)', 80: 'PINE SISKIN', 81: 'PINE WARBLER', 82: 'PURPLE FINCH (ADULT MALE)', 83: 'PYGMY NUTHATCH', 84: 'PYRRHULOXIA', 85: 'RED-BELLIED WOODPECKER', 86: 'RED-BREASTED NUTHATCH', 87: 'RED-WINGED BLACKBIRD (MALE)', 88: 'RING-NECKED PHEASANT (MALE)', 89: 'ROCK PIGEON', 90: 'RUBY-CROWNED KINGLET', 91: 'RUBY-THROATED HUMMINGBIRD (ADULT MALE)', 92: 'RUBY-THROATED HUMMINGBIRD (FEMALE, IMMATURE)', 93: 'RUFOUS HUMMINGBIRD (ADULT MALE)', 94: 'RUFOUS HUMMINGBIRD (FEMALE, IMMATURE)', 95: 'SONG SPARROW', 96: 'SPOTTED TOWHEE', 97: "STELLER'S JAY", 98: 'TUFTED TITMOUSE', 99: 'VARIED THRUSH', 100: 'WESTERN MEADOWLARK', 101: 'WHITE-BREASTED NUTHATCH', 102: 'WHITE-CROWNED SPARROW (ADULT)', 103: 'WHITE-CROWNED SPARROW (IMMATURE)', 104: 'WHITE-THROATED SPARROW (WHITE-STRIPED)', 105: 'WHITE-WINGED DOVE', 106: 'WILD TURKEY', 107: 'YELLOW-BELLIED SAPSUCKER', 108: "YELLOW-RUMPED WARBLER (BREEDING AUDUBON'S)", 109: 'YELLOW-RUMPED WARBLER (BREEDING MYRTLE)'}

        return labels

    def update_pred(self, predictions, confidence):
        predictions = predictions[0]
        top_five = {}
        predict_indexes = np.argpartition(predictions, -5)[-5:]
        indices = predict_indexes[np.argsort((-predictions)[predict_indexes])]

        for i in indices:
            print(self.labels[i])
            if self.labels[i] in self.pred_dict:
                self.pred_dict[self.labels[i]
                               ] += float(predictions[i] * confidence)
            else:
                self.pred_dict[self.labels[i]
                               ] = float(predictions[i] * confidence)


def main(_argv):
    """
    r = redis.Redis(host='localhost', port=6379, db=0)
    pipe = r.pubsub()
    pipe.subscribe({'birdEvents': my_handler})
    """
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(
        FLAGS)

    input_size = FLAGS.size
    saved_model_loaded = tf.saved_model.load(
        FLAGS.weights, tags=[tag_constants.SERVING])
    yolo_classifier = YoloClassifier(saved_model_loaded, input_size)
    species_classifier = SpeciesClassifier("../best_model_inception.hdf5")
    new_vid = True
    while True:
        # Poll for a new video and load it
        yolo_classifier.load_video("vid07-02-2020_08:29:08.avi")
        yolo_classifier.classify()

        species_classifier.classify(
            yolo_classifier.get_crops(), yolo_classifier.confidence_arr)
        # Save information about species in DB
        yolo_classifier.clear_data()
        species_classifier.reset_pred_dict()


"""
def my_handler(message):
    print('MY HANDLER: ', message['data'])

    yolo_classifier.load_video("../bird_recordings/vid"+message[data]+".avi")
    yolo_classifier.classify()

    species_classifier = SpeciesClassifier("../best_model_inception.hdf5")
    species_classifier.classify(
        yolo_classifier.get_crops(), yolo_classifier.confidence_arr)
"""

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
