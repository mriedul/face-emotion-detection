import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os

print("Testing Emotion Detection Model")

# Load model
model_path = 'models/best_emotion_model.keras'
if not os.path.exists(model_path):
    model_path = 'models/emotion_model_final.keras'

if not os.path.exists(model_path):
    print("ERROR: Model not found! Run train_model.py first.")
    exit()

model = keras.models.load_model(model_path)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def test_image(image_path):
    """Test model on a single image"""
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image from {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("No face detected in image!")
        return
    
    # Get first face
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = face_roi.astype('float32') / 255.0
    face_roi = np.expand_dims(face_roi, axis=0)
    face_roi = np.expand_dims(face_roi, axis=-1)
    
    # Predict
    prediction = model.predict(face_roi, verbose=0)
    emotion_idx = np.argmax(prediction)
    emotion = emotions[emotion_idx]
    confidence = prediction[0][emotion_idx] * 100
    
    # Display
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f"{emotion}: {confidence:.1f}%", (x, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Emotion: {emotion} ({confidence:.1f}%)")
    plt.axis('off')
    plt.show()
    
    # Show all predictions
    print("\nAll emotion probabilities:")
    for i, emo in enumerate(emotions):
        print(f"{emo:12s}: {prediction[0][i]*100:.2f}%")

# Example usage
if __name__ == "__main__":
    print("\nTo test on an image, call:")
    print("test_image('path/to/your/image.jpg')")
    print("\nExample:")
    print("test_image('screenshot_1.png')")