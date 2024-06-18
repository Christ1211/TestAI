from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from pyzbar import pyzbar

app = Flask(__name__)

detected_data = []

def decode(frame):
    global detected_data
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance detection
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find barcodes and QR codes
    decoded_objects = pyzbar.decode(thresh)
    for obj in decoded_objects:
        detected_text = obj.data.decode('utf-8')
        if detected_text not in detected_data:
            detected_data.append(detected_text)

        # Draw the bounding box around the detected QR code or barcode
        points = obj.polygon
        if len(points) > 4:  # If the points form a polygon (QR code)
            hull = cv2.convexHull(np.array([point for point in points if point is not None], dtype=np.float32))
            points = list(map(tuple, np.squeeze(hull)))
        else:
            points = [point for point in points if point is not None]
        
        for i in range(len(points)):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[(i+1) % len(points)][0]), int(points[(i+1) % len(points)][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

        # Put the detected text above the bounding box
        x, y, w, h = obj.rect
        cv2.putText(frame, detected_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def webcam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = decode(frame)

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/clear_data', methods=['GET'])
def clear_data():
    global detected_data
    detected_data = []
    return "Data cleared successfully"


@app.route('/')
def index():
    return render_template('indexQR.html')

@app.route('/video_feed')
def video_feed():
    return Response(webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_data')
def get_detected_data():
    global detected_data
    return jsonify(detected_data)

if __name__ == "__main__":
    app.run(debug=True)
