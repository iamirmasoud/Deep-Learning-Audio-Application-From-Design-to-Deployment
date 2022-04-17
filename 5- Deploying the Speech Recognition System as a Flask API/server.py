import os
import random

from flask import Flask, jsonify, request
from keyword_spotting_service import keyword_spotting_service

# instantiate flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to predict keyword

    :return (json): This endpoint returns a json file with the following format:
        {
            "keyword": "down"
        }
    """

    # get file from POST request and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # instantiate keyword spotting service singleton and get prediction
    kss = keyword_spotting_service()
    predicted_keyword = kss.predict(file_name)

    # we don't need the audio file anymore - let's delete it!
    os.remove(file_name)

    # send back result as a json file
    result = {"keyword": predicted_keyword}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)
