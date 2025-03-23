import cv2
from screeninfo import get_monitors

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened properly
if not cap.isOpened():
    print("Error. Could not open webcam.")
    exit()

# Get screen size (resolution)
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Set desired frame width and height to be proportional to the screen size 
frame_width = int(screen_width * 0.9)  # 90% of screen width
frame_height = int(screen_height * 0.9)  # 90% of screen height

# Set the webcam resolution to fit within the screen size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Create a window with the same size as the screen
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", frame_width, frame_height)

# Initialize the blur level (initial value)
blur_level = 30

# Function to set blur level dynamically
def set_blur(val):
    global blur_level
    blur_level = val

# Create trackbar for blur level adjustment
cv2.createTrackbar("Blur Level", "Output", blur_level, 100, set_blur)

while True:
    # Capture frame by frame
    check, frame = cap.read()

    # Check if frame is properly captured
    if not check:
        print("Error. Failed to capture frame")
        break

    # Convert frames to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect frontal faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Detect profile faces
    # profiles = profile_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Loop through all the detected faces
    for i, (x, y, w, h) in enumerate(faces):
        label = f"Face {i+1}"

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Extract ROI of faces
        face_region = frame[y:y + h, x:x + w]

        # Apply medianBlur to blur the face region with the dynamic blur level
        if blur_level % 2 == 0:  # Blur level is an odd number (for medianBlur)
            blur_level += 1
        blurred_face = cv2.medianBlur(face_region, blur_level)

        # Replace original faces with the blurred ones
        frame[y:y + h, x:x + w] = blurred_face

        # Add label above face
        label_y = max(y - 10, 10)
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)


    # Loop through all the detected profile faces
    # for i, (x, y, w, h) in enumerate(profiles):
    #     label = f"Face {i+1}"
  
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    #     profile_region = frame[y:y + h, x:x + w]

    #     if blur_level % 2 == 0:
    #         blur_level += 1
    #     blurred_profile = cv2.medianBlur(profile_region, blur_level)

    #     frame[y:y + h, x:x + w] = blurred_profile

    #     label_y = max(y - 10, 10)
    #     cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # Display the frame with blurred faces
    cv2.imshow('Output', frame)

    # 'Esc' to exit webcam feed
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

