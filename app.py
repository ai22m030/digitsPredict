import os
from consume_lib import predict
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_PATH'] = 'uploads'


@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file = os.path.join(app.config['UPLOAD_PATH'], filename)
        uploaded_file.save(file)
        return jsonify(prediction=predict(filename, "assignment01_model.h5", False).__str__())

    abort(500)
