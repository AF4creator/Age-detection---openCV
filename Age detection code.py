"""'pip install opencv-python opencv-python-headless numpy'-First type this code for install libraries for import
 In age_net and gender_net, change the path name "'C:/Users/username/Age detection/age_deploy.prototxt',
                                                   'C:/Users/username/Age detection/age_net.caffemodel'"

"""


import cv2
import numpy as np

# Load pre-trained models for face, age, and gender detection
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    age_net = cv2.dnn.readNetFromCaffe(
        'C:/Users/antof/Downloads/Age detection project/age_deploy.prototxt',
        'C:/Users/antof/Downloads/Age detection project/age_net.caffemodel'
    )
    gender_net = cv2.dnn.readNetFromCaffe(
        'C:/Users/antof/Downloads/Age detection project/gender_deploy.prototxt',
        'C:/Users/antof/Downloads/Age detection project/gender_net.caffemodel'
    )
    return face_cascade, age_net, gender_net

# Detect and predict age and gender
def process_frame(frame, face_cascade, age_net, gender_net, age_list, gender_list):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract face region
        face = frame[y:y + h, x:x + w]

        # Prepare face for deep learning models
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Display predictions
        label = f'Gender: {gender}, Age: {age}'
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return frame

# Start webcam and process frames
def start_webcam(face_cascade, age_net, gender_net, age_list, gender_list):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Process frame for age and gender detection
            frame = process_frame(frame, face_cascade, age_net, gender_net, age_list, gender_list)

            # Display the frame
            cv2.imshow('Age and Gender Detection', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Camera turned off and all windows closed.")

# Test multiple webcam indices
def test_webcam_indices():
    print("\nTesting additional webcam indices...")
    for index in range(3):  # Test up to 3 indices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Webcam found at index {index}.")
            ret, frame = cap.read()
            if ret:
                print("Frame read successfully.")
                cv2.imshow(f"Webcam Index {index}", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Error: Could not read frame at index {index}.")
            cap.release()
            break
        else:
            print(f"No webcam found at index {index}.")

# Main execution
if __name__ == "__main__":
    # Load models
    face_cascade, age_net, gender_net = load_models()

    # Define age and gender categories
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-80)']
    gender_list = ['Male', 'Female']

    # Start webcam for detection
    start_webcam(face_cascade, age_net, gender_net, age_list, gender_list)

    # Test additional webcam indices
    test_webcam_indices()

def start_webcam(face_cascade, age_net, gender_net, age_list, gender_list):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Process frame for age and gender detection
            frame = process_frame(frame, face_cascade, age_net, gender_net, age_list, gender_list)

            # Display the frame
            cv2.imshow('Age and Gender Detection', frame)

            # Check if the 'q' key is pressed or the window is closed
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Age and Gender Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Camera turned off and all windows closed.")
