from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import uuid
from final_mlmodel import SteganalysisInference

app = Flask(__name__, static_folder='assets', template_folder='.')

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model once
MODEL_PATH = "C:\\Users\\Aniruddh Rajagopal\\Downloads\\best_steganalysis_model\\best_steganalysis_model.joblib"
inferencer = SteganalysisInference(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

# Serve static files
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save file to upload folder
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        result = inferencer.predict_steganography(filepath)

        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Could not process image'}), 500

    return jsonify({'error': 'Unexpected error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
