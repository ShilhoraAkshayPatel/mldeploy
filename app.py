from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from flask_cors import CORS

from oauth2client.client import GoogleCredentials
import googleapiclient.discovery

# Change this values to match your project
PROJECT_ID = "deckgltest-1579098721098"
MODEL_NAME = "test"
CREDENTIALS_FILE = "credentials.json"

# These are the values we want a prediction for


def predictiohelper(input):
    # Connect to the Google Cloud-ML Service
    credentials = GoogleCredentials.from_stream(CREDENTIALS_FILE)
    service = googleapiclient.discovery.build(
        'ml', 'v1', credentials=credentials)

    inputs_for_prediction = input
    # Connect to our Prediction Model
    name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
    response = service.projects().predict(
        name=name, body={'instances': inputs_for_prediction}).execute()

    # Report any errors
    if 'error' in response:
        raise RuntimeError(response['error'])

    # Grab the results from the response object
    # results = response['predictions']

    results = response['predictions']
    return results
    # Print the results!


app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def index():
    return "go to /predict to get predction with json data as input"


@app.route('/api/predict', methods=['POST'])
def predict():
    predclass = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    data = request.get_json(force=True)
    predictions = predictiohelper(data)
    # print('INFO Predictions: {}'.format(predictions))
    return jsonify(predclass[np.argmax(predictions[0]["dense_5"])])
    # return jsonify(data)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port)
