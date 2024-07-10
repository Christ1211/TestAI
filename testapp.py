import base64
import cv2
from flask import Flask, request, jsonify
import torch
import os
import numpy as np
from flask_cors import CORS
from PIL import Image
import io
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

# Define paths
UPLOAD_FOLDER_IMAGES = 'dataset/val/images'
UPLOAD_FOLDER_LABELS = 'dataset/val/labels'
DATASET_FOLDER_IMAGES_TRAIN = 'dataset/images/train'
DATASET_FOLDER_IMAGES_VAL = 'dataset/images/val'
DATASET_FOLDER_LABELS_TRAIN = 'dataset/labels/train'
DATASET_FOLDER_LABELS_VAL = 'dataset/labels/val'

os.makedirs(UPLOAD_FOLDER_IMAGES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_LABELS, exist_ok=True)
os.makedirs(DATASET_FOLDER_IMAGES_TRAIN, exist_ok=True)
os.makedirs(DATASET_FOLDER_LABELS_TRAIN, exist_ok=True)

# Load class names and models
class_names_path = 'class_names.txt'
with open(class_names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

model_paths = {}
base_path = 'runs/train/'

for class_name in class_names:
    directory = f"{base_path}{class_name}"
    if os.path.exists(directory):
        model_paths[class_name] = f"{directory}/weights/Best.pt"
    else:
        print(f"Directory not found: {directory}, skipping...")

models = {}
for key, path in model_paths.items():
    try:
        models[key] = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
        print(f"Successfully loaded model for {key}")
    except Exception as e:
        print(f"Error loading model for {key}: {e}")

def transform_image(image_bytes):
    """Transform image to the format your model expects"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Use 640x640 for YOLOv5
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transformed_image = transform(image).unsqueeze(0)
    print(f"Transformed image shape: {transformed_image.shape}")  # Debug print
    return transformed_image

def detect_objects(image, models):
    highest_confidence = 0.0
    best_model = None
    results_dict = []

    for key, model in models.items():
        output = model(image)
        print(f"Output for model {key}: {output}")  # Debug print
        conf = output[0][:, 4].max().item() if len(output[0]) > 0 else 0.0
        if conf > highest_confidence:
            highest_confidence = conf
            best_model = model

    if best_model:
        output = best_model(image)
        print(f"Best model output: {output}")  # Debug print
        predictions = output[0]  # Get the first batch's predictions
        boxes = predictions[:, :4].cpu().numpy()  # Extract bounding box coordinates
        scores = predictions[:, 4].cpu().numpy()  # Extract confidence scores
        labels = predictions[:, 5].cpu().numpy()  # Extract class labels

        for i in range(len(boxes)):
            print(f"Box: {boxes[i]}, Score: {scores[i]}, Label: {labels[i]}")  # Debug print
            if scores[i] > 0.2:  # Temporarily lower confidence threshold
                label = f"{best_model.names[int(labels[i])]} {scores[i]:.2f}"
                results_dict.append({
                    "label": label,
                    "confidence": float(scores[i]),
                    "coordinates": boxes[i].tolist()
                })

    return results_dict

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        input_tensor = transform_image(image_bytes)
        results = detect_objects(input_tensor, models)
        return jsonify({'predictions': results})
    except Exception as e:
        print(f'Error: {e}')  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
