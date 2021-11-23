from flask import Flask, request
from model import train_model, predict_binary

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    train_model(learning_rate=data['learning_rate'], epochs=data['epochs'])
    return {'result': 'trained'}

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.get_data()
    return {'prediction': int(predict_binary(image_data)) }
