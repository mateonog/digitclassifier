from random import randrange
import os
import tensorflow as tf
from dataset import load_dataset

def save_test_images():
    _, _, x_test, _ = load_dataset()
    max_images = 10
    max_examples = x_test.shape[0]

    if not os.path.exists(os.path.join(os.getcwd(), 'images')):
        os.mkdir('images')

    for _ in range(max_images):
        n = randrange(0,max_examples)
        tf.keras.utils.save_img(os.path.join(os.getcwd(), 'images', f'{n}.jpg'), x_test[n])
