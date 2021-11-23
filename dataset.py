import tensorflow as tf
import numpy as np
from PIL import Image
import io

def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = process_train(x_train)
    x_test = process_train(x_test)

    y_train = one_hot_labels(y_train)
    y_test = one_hot_labels(y_test)

    return x_train, y_train, x_test, y_test

def one_hot_labels(y):
    return tf.keras.utils.to_categorical(y , num_classes=10)

def process_train(x):
    x = np.expand_dims(x, axis=-1)
    x = np.repeat(x, 3, axis=-1)
    x = tf.image.resize(x, [28,28])
    return x

def image_to_tensor(binary_image):
    image = Image.open(io.BytesIO(binary_image))
    return tf.keras.preprocessing.image.img_to_array(image)
