import cv2
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

# Load Haar cascade classifiers for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set default blur level
blur_level = 30

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_blur', methods=['POST'])
def update_blur():
    """ Update the blur level from the web slider """
    global blur_level
    blur_level = int(request.form['blur'])
    return jsonify(success=True, blur_level=blur_level)

def generate_frames():
    """ Capture video frames, detect faces, blur them, and stream the result """
    global blur_level
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            face_region = frame[y:y+h, x:x+w]
            
            # Ensure blur_level is an odd number (required for medianBlur)
            blur_value = blur_level if blur_level % 2 == 1 else blur_level + 1
            
            # Apply blurring
            blurred_face = cv2.medianBlur(face_region, blur_value)
            frame[y:y+h, x:x+w] = blurred_face
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Convert frame to JPEG and stream it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """ Video streaming route """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
