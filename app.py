from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import base64
from face_utils import FaceRecognitionSystem

app = Flask(__name__)

# Initialize camera
camera = cv2.VideoCapture(0)

# Initialize face recognition system
face_system = FaceRecognitionSystem()

# Function for streaming video with face recognition
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to RGB for recognition
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Recognize face(s)
            recognized, name, confidence, additional_info = face_system.recognize_face(frame_rgb)

            # Draw recognition info on frame if face was recognized
            if recognized:
                confidence = round(float(confidence), 2)
                label = f'{name} ({confidence}%)'
                cv2.putText(frame, label, (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Optional: draw bounding box if your system provides it
                if 'box' in additional_info:
                    x, y, w, h = additional_info['box']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    if not face_system or face_system.model is None:
        return render_template('index.html', error="Model failed to load")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        success, frame = camera.read()
        if not success:
            return jsonify({"error": "Failed to capture image"}), 500

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        recognized, name, confidence, additional_info = face_system.recognize_face(frame_rgb)

        confidence = round(float(confidence), 2)
        response_data = {
            "recognized": recognized,
            "name": name if recognized else None,
            "confidence": confidence,
            "info": additional_info if recognized else None
        }

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name")
    additional_info = data.get("info", {})

    if not name:
        return jsonify({"error": "Name is required"}), 400

    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image"}), 500

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    success, message = face_system.register_new_face(frame_rgb, name, additional_info)

    if success:
        return jsonify({"message": message})
    else:
        return jsonify({"error": message}), 400

@app.route("/capture_for_register", methods=["POST"])
def capture_for_register():
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image"}), 500

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return jsonify({"message": "Image captured for registration"}), 200

@app.route("/test_pipeline", methods=["GET"])
def test_pipeline():
    test_image_path = "../test_image.png"
    image = cv2.imread(test_image_path)

    if image is None:
        return jsonify({"error": f"Couldn't load image from path: {test_image_path}"}), 400

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    recognized, name, confidence, info = face_system.recognize_face(image_rgb)

    return jsonify({
        "recognized": recognized,
        "name": name if recognized else None,
        "confidence": round(float(confidence), 2),
        "info": info if recognized else None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
