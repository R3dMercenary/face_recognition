from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import base64
from face_utils import face_system

app = Flask(__name__)

# Initialize camera
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
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

@app.route('/capture_for_register', methods=['POST'])
def capture_for_register():
    if face_system is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image"}), 500
    
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "image": encoded_image
    })

@app.route('/recognize', methods=['POST'])
def recognize():
    if not face_system or not face_system.model:
        return jsonify({"error": "Model not loaded"}), 500
        
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image"}), 500
    
    # Process the image
    recognized, name, confidence, additional_info = face_system.recognize_face(frame)
    
    # Encode the image
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    if recognized:
        return jsonify({
            "recognized": True,
            "name": name,
            "confidence": confidence,
            "additional_info": additional_info,
            "image": encoded_image
        })
    else:
        return jsonify({
            "recognized": False,
            "image": encoded_image
        })
    
@app.route('/register', methods=['POST'])
def register():
    if face_system is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    name = data.get('name')
    additional_info = data.get('additional_info', '')
    
    if not name:
        return jsonify({"error": "Name is required"}), 400
    
    # Capture current frame
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image"}), 500
    
    # Register the face
    success, message = face_system.register_new_face(frame, name, additional_info)
    
    if success:
        return jsonify({
            "success": True,
            "message": message
        })
    else:
        return jsonify({
            "error": message
        }), 400

@app.route('/test_pipeline')
def test_pipeline():
    if not face_system or not face_system.model:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    
    try:
        # Create test image
        test_img = np.random.rand(480, 640, 3) * 255
        test_img = test_img.astype('uint8')
        
        # Test registration
        reg_status, reg_msg = face_system.register_new_face(test_img, "Test User", "Test Info")
        if not reg_status:
            return jsonify({"status": "error", "message": f"Registration failed: {reg_msg}"}), 500
        
        # Test recognition
        recognized, name, confidence, _ = face_system.recognize_face(test_img)
        
        return jsonify({
            "status": "success",
            "registration": "success",
            "recognition": {
                "recognized": recognized,
                "name": name,
                "confidence": confidence
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
@app.route('/check_faces')
def check_faces():
    if not face_system:
        return jsonify({"error": "System not loaded"}), 500
        
    return jsonify({
        "loaded_faces": face_system.known_face_names,
        "count": len(face_system.known_face_encodings)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)