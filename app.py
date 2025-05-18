from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import face_recognition
import io
import os
from datetime import datetime
from PIL import Image
import pickle

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'detected_faces'
ENCODING_FILE = 'encodings.pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve main pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

# Endpoint to serve saved face images (optional)
@app.route('/detected_faces/<filename>')
def get_saved_face(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# API endpoint: Detect and save face + encoding
@app.route('/api/detect', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    image = face_recognition.load_image_file(io.BytesIO(img_bytes))

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return jsonify({'message': '❌ No face detected'}), 200

    pil_image = Image.fromarray(image)

    # Load existing encodings
    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE, 'rb') as f:
            known_faces = pickle.load(f)
    else:
        known_faces = []

    saved_faces = []

    for i, (top, right, bottom, left) in enumerate(face_locations):
        face_image = pil_image.crop((left, top, right, bottom))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"face_{timestamp}_{i}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        face_image.save(save_path)
        saved_faces.append(filename)

        encoding = face_recognition.face_encodings(image, known_face_locations=[(top, right, bottom, left)])[0]

        known_faces.append({
            'filename': filename,
            'encoding': encoding
        })

    # Save updated encodings
    with open(ENCODING_FILE, 'wb') as f:
        pickle.dump(known_faces, f)

    return jsonify({
        'message': f'✅ {len(face_locations)} face(s) detected',
        'saved_files': saved_faces
    }), 200

# ✅ Face matching route
@app.route('/api/match', methods=['POST'])
def match_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    image = face_recognition.load_image_file(io.BytesIO(img_bytes))

    uploaded_face_encodings = face_recognition.face_encodings(image)

    if len(uploaded_face_encodings) == 0:
        return jsonify({'message': '❌ No face detected in uploaded image'}), 200

    uploaded_encoding = uploaded_face_encodings[0]

    if not os.path.exists(ENCODING_FILE):
        return jsonify({'message': '⚠️ No known faces to match against'}), 200

    with open(ENCODING_FILE, 'rb') as f:
        known_faces = pickle.load(f)

    matches = []
    for known in known_faces:
        match = face_recognition.compare_faces([known['encoding']], uploaded_encoding, tolerance=0.4)[0]
        print(f"Comparing with {known['filename']}, Match: {match}")
    if match:
        matches.append(known['filename'])

    if matches:
        return jsonify({
            'message': f'✅ Match found with {len(matches)} saved face(s)',
            'matched_files': matches
        }), 200
    else:
        return jsonify({'message': '❌ No match found'}), 200

if __name__ == '__main__':
    app.run(debug=True)