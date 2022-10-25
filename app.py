import os
import uuid
from consume_lib import predict
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_PATH'] = 'uploads'
app.config['PREDICT_PATH'] = 'predictions'


@app.route('/swe/numberprediction/rest/data/v1.0/json/en/assignment01_model/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file = os.path.join(app.config['UPLOAD_PATH'], filename)
        predict_file_name = uuid.uuid4().__str__()
        predict_file_path = os.path.join(app.config['PREDICT_PATH'], predict_file_name)
        uploaded_file.save(file)

        predict_file = open(predict_file_path, 'w')
        predict_file.write(predict(filename, "assignment01_model.h5", False).__str__())

        return jsonify(file_name=predict_file_name)

    abort(500)


@app.route('/swe/numberprediction/rest/data/v1.0/json/en/assignment01_model/', methods=['GET'])
def prediction():
    predict_file_name = request.args.get('file')
    filename = secure_filename(predict_file_name)
    if filename != '':
        file = os.path.join(app.config['PREDICT_PATH'], filename)
        predict_file = open(file, 'r')
        return jsonify(prediction=predict_file.readline())

    abort(500)
