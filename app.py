from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
import torch
import cv2
import os
import shutil
import subprocess
import logging
import yaml  # Import the yaml module
import base64
import json
import time
import numpy as np

from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define paths for uploads and dataset
UPLOAD_FOLDER_IMAGES = 'dataset/val/images'
UPLOAD_FOLDER_LABELS = 'dataset/val/labels'
DATASET_FOLDER_IMAGES_TRAIN = 'dataset/images/train'
DATASET_FOLDER_LABELS_TRAIN = 'dataset/labels/train'
os.makedirs(UPLOAD_FOLDER_IMAGES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_LABELS, exist_ok=True)
os.makedirs(DATASET_FOLDER_IMAGES_TRAIN, exist_ok=True)
os.makedirs(DATASET_FOLDER_LABELS_TRAIN, exist_ok=True)

# Example script

# Path to your class names file
file_path = 'class_names.txt'

# Initialize an empty list to store class names
class_names = []

# Read class names from the text file
with open(file_path, 'r') as file:
    for line in file:
        # Remove any extra whitespace, like newline characters
        class_names.append(line.strip())

# Now class_names list contains your class names

# Create a dictionary to map indices to class names
names_dict = {i: class_names[i] for i in range(len(class_names))}

# Print the dictionary for verification
print("Names dictionary:")
print(names_dict)

# Example yaml structure
yaml_data = {
    'train': "dataset/images/train/",  # Path to training images with class names
    'val': "dataset/images/train/",    # Path to validation images with class names
    'names': names_dict,
    # Add other YOLOv5 parameters here as needed
}

# Example usage: You might write this yaml_data to a yaml file for YOLOv5 training
# Here's how you might write it using PyYAML
import yaml

yaml_file_path = 'krewkrew2.yaml'
with open(yaml_file_path, 'w') as yamlfile:
    yaml.safe_dump(yaml_data, yamlfile, default_flow_style=False)


  # For uploading
def load_model2(model_path='runs/train/exp7/weights/last.pt'):
    # Load the YOLOv5 model
   model2 = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
   return model2


def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model

# Load all models at the start
class_names_path = 'class_names.txt'
with open(class_names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Create the model_paths dictionary
model_paths = {}
base_path = 'runs/train/'

# Loop through the class names and construct the model paths
for class_name in class_names:
    directory = f"{base_path}{class_name}"
    if os.path.exists(directory):
        model_paths[class_name] = f"{directory}/weights/Best.pt"
    else:
        print(f"Directory not found: {directory}, skipping...")

# Print the modified model_paths
print(model_paths)

models = {key: load_model(path) for key, path in model_paths.items()}

# Function to generate frames using all models
def gen_frames(models):
    camera = cv2.VideoCapture(0)  # Use 0 for webcam
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to RGB as model expects RGB input
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for key, model in models.items():
                # Perform object detection using the loaded model
                results = model(frame_rgb)
                # Draw bounding boxes
                for result in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = result
                    if conf > 0.8:  # Check confidence threshold
                        label = f"{model.names[int(cls)]} {conf:.2f}"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red bounding box
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(models), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to read class names from the file
def read_class_names():
    with open('class_names.txt', 'r') as f:
        return [line.strip() for line in f.readlines()]

# Function to check folders
def check_folders(class_names):
    folder_status = {}
    for class_name in class_names:
        folder_path = f'runs/train/{class_name}'
        folder_status[class_name] = os.path.exists(folder_path)
    return folder_status

@app.route('/add_class', methods=['POST'])
def add_class():
    new_class_name = request.form.get('new_class_name')

    if new_class_name:
        # Append the new class name to class_names.txt
        with open('class_names.txt', 'a') as f:
            f.write(new_class_name + '\n')

    # Redirect or return a response as needed
    return redirect('/train')  # Redirect to main page or another appropriate location

@app.route('/train')
def train_page():
    class_names = read_class_names()  # Read class names dynamically
    folder_status = check_folders(class_names)  # Check folder status
    return render_template('train.html', class_names=class_names, folder_status=folder_status)

@app.route('/trained_models')
def trained_models():
    class_names = read_class_names()  # Read class names dynamically
    folder_status = check_folders(class_names)  # Check folder status again
    return render_template('train.html', class_names=class_names, folder_status=folder_status)

from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import shutil

from flask import request


@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    if 'images' not in request.files or 'class_name' not in request.form:
        return redirect(request.url)

    class_name = request.form['class_name']
    class_name2 = request.form['classline']
    images = request.files.getlist('images')

    if not images:
        return redirect(request.url)

    for image in images:
        if image.filename == '':
            continue

        # Save the uploaded image temporarily
        filename = secure_filename(image.filename)
        image_path = os.path.join(UPLOAD_FOLDER_IMAGES, filename)
        image.save(image_path)

        # Perform object detection using the loaded model to get bounding boxes
        model2 = load_model2()
        results = model2(image_path)
        labels = results.xyxy[0]

        if len(labels) == 0:
            # No objects detected, remove the temporary saved image
            os.remove(image_path)
            continue

        # Create the label file and annotate labels
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(UPLOAD_FOLDER_LABELS, label_filename)
        with open(label_path, 'w') as label_file:
            for *xyxy, conf, cls in labels:
                x_center = (xyxy[0].item() + xyxy[2].item()) / 2 / results.ims[0].shape[1]
                y_center = (xyxy[1].item() + xyxy[3].item()) / 2 / results.ims[0].shape[0]
                width = (xyxy[2].item() - xyxy[0].item()) / results.ims[0].shape[1]
                height = (xyxy[3].item() - xyxy[1].item()) / results.ims[0].shape[0]
                label_file.write(f"{class_name2} {x_center} {y_center} {width} {height}\n")

        # Copy the image and label file to the training dataset folder
        class_folder_images = os.path.join(DATASET_FOLDER_IMAGES_TRAIN, class_name)
        class_folder_labels = os.path.join(DATASET_FOLDER_LABELS_TRAIN, class_name)
        os.makedirs(class_folder_images, exist_ok=True)
        os.makedirs(class_folder_labels, exist_ok=True)
        shutil.copy(image_path, class_folder_images)
        shutil.copy(label_path, class_folder_labels)

    return redirect(url_for('train_page'))


@app.route('/start_training', methods=['POST'])
def start_training():
    try:
        epochs = int(request.form.get('epochs', 10))
        class_name = request.form.get('class_name')

        # Check GPU availability and set device
        if torch.cuda.is_available():
            print("GPU available. Using GPU for training.")
            device = torch.device("cuda")
        else:
            print("GPU not available. Using CPU for training.")
            device = torch.device("cpu")

        # Define the path to the YAML file for the selected class
        yaml_file_path = os.path.join('krewkrew2.yaml')

        # Update the YAML file with new paths
        train_path = os.path.join('dataset/images/train', class_name)
        val_path = os.path.join('dataset/images/train', class_name)
        
        with open(yaml_file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)

        yaml_data['train'] = train_path
        yaml_data['val'] = val_path

        with open(yaml_file_path, 'w') as file:
            yaml.safe_dump(yaml_data, file, default_flow_style=False)

        # Load model
        model_path = 'yolov5s.pt'  # Adjust this path as per your model location
        model = load_model(model_path).to(device)

        # Use subprocess to call the training script
        command = [
            'python', 'train.py',
            '--img', '640',
            '--batch', '16',
            '--epochs', str(epochs),
            '--data', yaml_file_path,  # Use updated YAML file
            '--weights', 'yolov5s.pt',
            '--name', class_name
        ]

        # Capture the stdout and stderr of the subprocess
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Log the stdout and stderr
        logging.info(result.stdout)
        if result.returncode != 0:
            logging.error(f"Training script failed with error: {result.stderr}")
            return "Training Failed"

        return "TRAINING FINISHED!"
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return f"Error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
@app.route('/dataset/val/images/<filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER_IMAGES, filename)


if __name__ == '__main__':
    app.run(debug=True)

