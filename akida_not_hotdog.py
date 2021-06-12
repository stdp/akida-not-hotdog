import time
import cv2
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from akida import Model

HOTDOG_FBZ = "models/hotdog.fbz"
HOTDOG_NEURON = 1
HOTDOGS_PER_SECOND = 2
CAMERA_SRC = 0
NUM_CLASSES = 2
TARGET_WIDTH = 224
TARGET_HEIGHT = 224


class Camera:
    def __init__(self):
        self.stream = VideoStream(src=CAMERA_SRC).start()

    def get_input_array(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        input_array = img_to_array(frame)
        input_array = np.array([input_array], dtype="uint8")
        return input_array

    def show_frame(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF


class Inference:
    def __init__(self, camera):
        # init the camera
        self.camera = camera
        self.hotdog_model = Model(filename=HOTDOG_FBZ)

    def infer(self):
        i = self.camera.get_input_array()
        p = self.hotdog_model.predict(i, num_classes=NUM_CLASSES)
        if p[0] == HOTDOG_NEURON:
            print("HOTDOG")
        else:
            print("NOT HOTDOG")


camera = Camera()
inference = Inference(camera)

while True:
    inference.infer()
    camera.show_frame()
    time.sleep(1 / HOTDOGS_PER_SECOND)
