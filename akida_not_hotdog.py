import time
import cv2
from threading import Timer
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from akida import Model
from pynput import keyboard


HOTDOG_FBZ = "models/hotdog.fbz"
HOTDOG_NEURON = 1

CAMERA_SRC = 0
NUM_CLASSES = 10
TARGET_WIDTH = 224
TARGET_HEIGHT = 224
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

TEXT_COLOUR = (255, 255, 255)

HOTDOG_LABEL = "HOTDOG"
NOTHOTDOG_LABEL = "NOT HOTDOG"


class Controls:

    def __init__(self, inference):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()
        self.inference = inference

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                self.inference.infer()
        except AttributeError:
            pass

    def on_release(self, key):
        if key == keyboard.Key.esc:
            return False


class Camera:
    def __init__(self):
        self.stream = VideoStream(src=CAMERA_SRC).start()
        self.label = ""

    def get_input_array(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        input_array = img_to_array(frame)
        input_array = np.array([input_array], dtype="uint8")
        return input_array

    def show_frame(self):
        frame = cv2.resize(self.stream.read(), (FRAME_WIDTH, FRAME_HEIGHT))
        frame = self.label_frame(frame)
        cv2.imshow("Akida Not Hotdog", frame)
        key = cv2.waitKey(1) & 0xFF

    def label_frame(self, frame):
        frame = cv2.putText(
            frame,
            str(self.label),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            TEXT_COLOUR,
            4,
            cv2.LINE_AA,
        )
        return frame

    def set_label(self, label):
        self.label = label


class Inference:
    def __init__(self, camera):
        # init the camera
        self.camera = camera
        self.hotdog_model = Model(filename=HOTDOG_FBZ)

    def infer(self):
        i = self.camera.get_input_array()
        p = self.hotdog_model.predict(i, num_classes=NUM_CLASSES)
        if p[0] == HOTDOG_NEURON:
            print(HOTDOG_LABEL)
            self.camera.set_label(HOTDOG_LABEL)
        else:
            print(NOTHOTDOG_LABEL)
            self.camera.set_label(NOTHOTDOG_LABEL)


camera = Camera()
inference = Inference(camera)
controls = Controls(inference)

while True:
    camera.show_frame()
