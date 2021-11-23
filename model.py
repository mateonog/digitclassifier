import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from dataset import load_dataset, image_to_tensor

model = None

def train_model(learning_rate=0.001, epochs = 1):
    x_train, y_train, x_test, y_test = load_dataset()
    compiled_model = compile_model(learning_rate=learning_rate)
    compiled_model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose = 2)

    global model
    model = compiled_model

def predict_example(example):
    return tf.math.argmax(model.predict(tf.reshape(example, (1, 28, 28, 3)))[0], axis = 0)

def predict_binary(binary_image):
    return predict_example(image_to_tensor(binary_image)).numpy()

def compile_model(learning_rate):
    model_input = Input(shape=(28,28,3))
    resnet = ResNet50(weights='imagenet', include_top = False, input_tensor = model_input)
    gap = GlobalMaxPooling2D()(resnet.output)
    output = Dense(10, activation='softmax', use_bias=True)(gap)

    model = Model(resnet.input, output)
    model.compile(loss = CategoricalCrossentropy(), metrics = CategoricalAccuracy(), optimizer = Adam(learning_rate=learning_rate))
    return model
