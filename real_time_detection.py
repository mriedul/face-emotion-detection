import cv2
import numpy as np
from tensorflow import keras
import os

print("="*60)
print("REAL-TIME EMOTION DETECTION")
print("="*60)

# Load trained model
model_path = 'models/best_emotion_model.keras'
if not os.path.exists(model_path):
    model_path = 'models/emotion_model_final.keras'

if not os.path.exists(model_path):
    print("\nERROR: Model not found!")
    print("Please run train_model.py first to train the model.")
    exit()

print(f"\nLoading model from: {model_path}")
model = keras.models.load_model(model_path)
print("Model loaded successfully!")

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
print("\nInitializing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot access webcam!")
    exit()

print("\nWebcam ready!")
print("\nINSTRUCTIONS:")
print("- Make different facial expressions")
print("- Press 'q' to quit")
print("- Press 's' to save screenshot")
print("\nStarting emotion detection...\n")

frame_count = 0
screenshot_count = 0
last_emotion = "Unknown"
last_confidence = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Process every 2 frames for speed
    if frame_count % 2 == 0:
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            # Predict emotion
            prediction = model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(prediction)
            last_emotion = emotions[emotion_idx]
            last_confidence = prediction[0][emotion_idx] * 100
    
    # Draw on all frames
    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Emotion label
        if frame_count >= 2:
            label = f"{last_emotion}: {last_confidence:.1f}%"
            
            # Background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(frame, (x, y-40), (x + text_size[0] + 10, y), (0, 255, 0), -1)
            
            # Emotion text
            cv2.putText(frame, label, (x+5, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Display info
    cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show frame
    cv2.imshow('Emotion Detection - My Deep Learning Project', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('s'):
        screenshot_count += 1
        filename = f'screenshot_{screenshot_count}.png'
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\nEmotion detection stopped.")
print("Thank you!")