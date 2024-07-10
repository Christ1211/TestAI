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
import random
import ssl
from flask_cors import CORS
from PIL import Image
import io
import torchvision.transforms as transforms  # Correct import

from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Define paths for uploads and dataset
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

# Example script

# Path to your class names file
file_path = 'class_names.txt'

# Initialize an empty list to store class names
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
    model_path = f"{directory}/weights/best.pt"
    if os.path.exists(directory):
        if os.path.exists(model_path):
            model_paths[class_name] = model_path
        else:
            print(f"File not found: {model_path}, deleting directory...")
            shutil.rmtree(directory)
    else:
        print(f"Directory not found: {directory}, skipping...")

# Print the modified model_paths
print(model_paths)

def load_model(path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=path)

models = {key: load_model(path) for key, path in model_paths.items()}

def process_frame(frame, models):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for key, model in models.items():
        results = model(frame_rgb)
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result
            if conf > 0.8:
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame

@app.route('/video_feed', methods=['POST'])
def video_feed():
    data = request.json['frame']
    img_data = base64.b64decode(data)
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    processed_frame = process_frame(frame, models)

    ret, buffer = cv2.imencode('.jpg', processed_frame)
    frame = buffer.tobytes()
    
    return Response(frame, mimetype='image/jpeg')

@app.route('/')
def index():
    return render_template('index.html')



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

    temp_images = []
    temp_labels = []

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

        temp_images.append(image_path)
        temp_labels.append(label_path)

    # Shuffle and split the data into training and validation sets
    combined = list(zip(temp_images, temp_labels))
    random.shuffle(combined)
    temp_images[:], temp_labels[:] = zip(*combined)

    split_idx = int(0.7 * len(temp_images))
    train_images = temp_images[:split_idx]
    val_images = temp_images[split_idx:]
    train_labels = temp_labels[:split_idx]
    val_labels = temp_labels[split_idx:]

    # Move files to the appropriate directories
    for img, lbl in zip(train_images, train_labels):
        class_folder_images_train = os.path.join(DATASET_FOLDER_IMAGES_TRAIN, class_name)
        class_folder_labels_train = os.path.join(DATASET_FOLDER_LABELS_TRAIN, class_name)
        os.makedirs(class_folder_images_train, exist_ok=True)
        os.makedirs(class_folder_labels_train, exist_ok=True)
        shutil.move(img, class_folder_images_train)
        shutil.move(lbl, class_folder_labels_train)

    for img, lbl in zip(val_images, val_labels):
        class_folder_images_val = os.path.join(DATASET_FOLDER_IMAGES_VAL, class_name)
        class_folder_labels_val = os.path.join(DATASET_FOLDER_LABELS_VAL, class_name)
        os.makedirs(class_folder_images_val, exist_ok=True)
        os.makedirs(class_folder_labels_val, exist_ok=True)
        shutil.move(img, class_folder_images_val)
        shutil.move(lbl, class_folder_labels_val)

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
        yaml_file_path = 'krewkrew2.yaml'

        # Update the YAML file with new paths
        train_path = os.path.join('dataset/images/train', class_name)
        val_path = os.path.join('dataset/images/val', class_name)
        
        with open(yaml_file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)

        yaml_data['train'] = train_path
        yaml_data['val'] = val_path

        with open(yaml_file_path, 'w') as file:
            yaml.safe_dump(yaml_data, file, default_flow_style=False)

        # Define the path where the weights are expected to be stored
        weights_dir = 'runs/train'  # This is the default directory where YOLOv5 stores trained weights
        class_weights_dir = os.path.join(weights_dir, class_name)
        weights_file = os.path.join(class_weights_dir, 'weights/best.pt')
        
        # Check if the weights file for the class exists
        if os.path.isfile(weights_file):
            weights_path = weights_file
            print(f"Using existing weights: {weights_path}")
        else:
            weights_path = 'yolov5s.pt'  # Default YOLOv5 weights
            print(f"No existing weights found. Using default weights: {weights_path}")

        # Ensure the directory exists
        if not os.path.exists(class_weights_dir):
            os.makedirs(class_weights_dir)

        # Load model
        model = load_model(weights_path).to(device)

        # Use subprocess to call the training script
        command = [
            'python', 'train.py',
            '--img', '640',
            '--batch', '20',
            '--epochs', str(epochs),
            '--data', yaml_file_path,  # Use updated YAML file
            '--weights', weights_path,
            '--name', class_name,
            '--project', weights_dir,  # Ensure training outputs are in the correct directory
            '--exist-ok'  # Allow existing directory to be used
        ]

        # Capture the stdout and stderr of the subprocess
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Log the stdout and stderr
        logging.info(result.stdout)
        if result.returncode != 0:
            logging.error(f"Training script failed with error: {result.stderr}")
            return "Training Failed"

        # Convert the trained model to TensorFlow.js format
        tfjs_output_dir = os.path.join(class_weights_dir, 'tfjs_model')
        if not os.path.exists(tfjs_output_dir):
            os.makedirs(tfjs_output_dir)
        
        # Assuming you have the model conversion script `pt_to_tfjs.py`
        conversion_command = [
            'python', 'pt_to_tfjs.py',
            '--model', weights_file,
            '--output', tfjs_output_dir
        ]
        
        conversion_result = subprocess.run(conversion_command, capture_output=True, text=True)
        logging.info(conversion_result.stdout)
        if conversion_result.returncode != 0:
            logging.error(f"Model conversion failed with error: {conversion_result.stderr}")
            return "Model Conversion Failed"

        return "TRAINING AND CONVERSION FINISHED!"
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return f"Error occurred: {str(e)}"


#levie code

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)

        highest_confidence = 0.0
        best_model = None
        results_dict = []

        for key, model in models.items():
            output = model(input_tensor)
            conf = output[0][:, 4].max().item() if len(output[0]) > 0 else 0.0
            if conf > highest_confidence:
                highest_confidence = conf
                best_model = model

        if best_model:
            output = best_model(input_tensor)
            predictions = output[0]
            boxes = predictions[:, :4].cpu().numpy()
            scores = predictions[:, 4].cpu().numpy()
            labels = predictions[:, 5].cpu().numpy()

            for i in range(len(boxes)):
                if scores[i] > 0.2:
                    label = f"{best_model.names[int(labels[i])]} {scores[i]:.2f}"
                    results_dict.append({
                        "label": label,
                        "confidence": float(scores[i]),
                        "coordinates": boxes[i].tolist()
                    })

        return jsonify({'predictions': results_dict})
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500




context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain('certificate.pem', 'key.pem')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1501, debug=True, ssl_context=context)


