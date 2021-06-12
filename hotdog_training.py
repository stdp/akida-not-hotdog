import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from akida_models import mobilenet_edge_imagenet_pretrained
from cnn2snn import convert
from akida import FullyConnected

HOTDOG = "img/hotdog.jpg"
NOT_HOTDOG = "img/not_hotdog.jpg"
HOTDOG_FBZ = "models/hotdog.fbz"
NUM_NEURONS_PER_CLASS = 1
HOTDOG_NEURON = 1
NOT_HOTDOG_NEURON = 0
NUM_WEIGHTS = 350
NUM_CLASSES = 2
TARGET_WIDTH = 224
TARGET_HEIGHT = 224
TARGET_CHANNELS = 3

# build Akida model from mobilenet
model_keras = mobilenet_edge_imagenet_pretrained()
model_ak = convert(model_keras, input_scaling=(128, 128))
model_ak.pop_layer()
layer_fc = FullyConnected(
    name="akida_edge_layer",
    num_neurons=NUM_CLASSES * NUM_NEURONS_PER_CLASS,
    activations_enabled=False,
)
model_ak.add(layer_fc)
model_ak.compile(
    num_weights=NUM_WEIGHTS, num_classes=NUM_CLASSES, learning_competition=0.1
)

# learn hotdog
image = load_img(
    HOTDOG, target_size=(TARGET_WIDTH, TARGET_HEIGHT), color_mode="rgb"
)
hotdog_array = img_to_array(image)
hotdog_array = np.array([hotdog_array], dtype="uint8")
model_ak.fit(hotdog_array, HOTDOG_NEURON)

# learn not hotdog
image = load_img(
    NOT_HOTDOG, target_size=(TARGET_WIDTH, TARGET_HEIGHT), color_mode="rgb"
)
hotdog_array = img_to_array(image)
hotdog_array = np.array([hotdog_array], dtype="uint8")
model_ak.fit(hotdog_array, NOT_HOTDOG_NEURON)

# save hotdog model
model_ak.save(HOTDOG_FBZ)
