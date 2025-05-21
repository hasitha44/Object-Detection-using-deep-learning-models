from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import torch
import cv2
import numpy as np
import base64
import os
from werkzeug.utils import secure_filename
import json  # Import the json module

app = Flask(__name__)

# Configure upload folder (optional)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

def process_frame(frame):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(pil_img)
    predictions = results.pandas().xyxy[0].to_dict(orient='records')
    return frame, predictions

def draw_predictions_on_frame(frame, predictions):
    img_cv = frame.copy()
    for pred in predictions:
        x1, y1, x2, y2 = int(pred['xmin']), int(pred['ymin']), int(pred['xmax']), int(pred['ymax'])
        label = f"{pred['name']} {pred['confidence']:.2f}"
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    _, img_encoded = cv2.imencode('.jpg', img_cv)
    base64_image = base64.b64encode(img_encoded).decode('utf-8')
    return base64_image

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    results = model(img)
    predictions = results.pandas().xyxy[0].to_dict(orient='records')
    return jsonify({'predictions': predictions})

@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'error': 'Could not open video'}), 500

    processed_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, predictions = process_frame(frame)
        processed_frame_base64 = draw_predictions_on_frame(frame, predictions)
        processed_frames.append(processed_frame_base64)

    cap.release()
    os.remove(video_path) # Clean up uploaded video

    return jsonify({'processed_frames': processed_frames})

@app.route('/process_webcam_frame', methods=['POST'])
def process_webcam_frame():
    try:
        data = json.loads(request.data)
        base64_image = data.get('image')
        if not base64_image:
            return jsonify({'error': 'No image data received'}), 400

        img_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_with_predictions, predictions = process_frame(frame)
        processed_frame_base64 = draw_predictions_on_frame(frame_with_predictions, predictions)

        return jsonify({'processed_frame': processed_frame_base64})

    except Exception as e:
        print(f"Error processing webcam frame: {e}")
        return jsonify({'error': 'Error processing webcam frame'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')