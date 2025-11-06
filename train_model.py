import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

print("=" * 60)
print("EMOTION DETECTION MODEL TRAINING")
print("=" * 60)

# Step 1: Download Dataset
print("\n[1/6] Downloading FER2013 dataset...")
path = "fer2013"
print(f"Dataset downloaded at: {path}")

# Setup paths
train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')

# Emotion classes
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(emotions)

print(f"\nEmotions to detect: {emotions}")

# Step 2: Data Exploration
print("\n[2/6] Exploring dataset...")
print("\nTraining data distribution:")
train_counts = {}
for emotion in emotions:
    emotion_path = os.path.join(train_dir, emotion)
    count = len(os.listdir(emotion_path))
    train_counts[emotion] = count
    print(f"  {emotion}: {count} images")

# Visualize distribution
plt.figure(figsize=(10, 5))
plt.bar(train_counts.keys(), train_counts.values(), color='skyblue')
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.title('Training Data Distribution', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data_distribution.png')
print("\nData distribution plot saved as 'data_distribution.png'")
plt.close()

# Step 3: Setup Data Generators
print("\n[3/6] Setting up data augmentation...")

img_size = 48
batch_size = 128

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.15
)

# Test data (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print(f"\nDataset split:")
print(f"  Training: {train_gen.samples} images")
print(f"  Validation: {val_gen.samples} images")
print(f"  Testing: {test_gen.samples} images")

# Step 4: Build Model
print("\n[4/6] Building CNN model...")

def build_emotion_model():
    """Build efficient CNN for emotion detection"""
    
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Fully Connected
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

model = build_emotion_model()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Step 5: Train Model
print("\n[5/6] Training model...")

# Create models folder if doesn't exist
os.makedirs('models', exist_ok=True)

# Callbacks
checkpoint = ModelCheckpoint(
    'models/best_emotion_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stop, reduce_lr]

# Train
print("\nStarting training... (will stop automatically when optimal)")
epochs = 15

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

print(f"\nTraining completed! Stopped at epoch: {len(history.history['loss'])}")

# Step 6: Evaluate and Visualize
print("\n[6/6] Evaluating model...")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Training', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax2.set_title('Model Loss', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png')
print("\nTraining curves saved as 'training_history.png'")
plt.close()

# Test evaluation
print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\n{'='*60}")
print(f"TEST RESULTS")
print(f"{'='*60}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Confusion Matrix
print("\nGenerating confusion matrix...")
test_gen.reset()
predictions = model.predict(test_gen, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotions, yticklabels=emotions)
plt.title('Confusion Matrix - Emotion Detection', fontsize=14)
plt.ylabel('True Emotion', fontsize=12)
plt.xlabel('Predicted Emotion', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.close()

# Classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred, target_names=emotions))

# Per-class accuracy
print("\nPer-Class Accuracy:")
print("-"*60)
class_acc = cm.diagonal() / cm.sum(axis=1)
for i, emotion in enumerate(emotions):
    print(f"{emotion.capitalize():12s}: {class_acc[i]*100:.2f}%")

# Save final model
model.save('models/emotion_model_final.keras')
print("\n" + "="*60)
print("MODEL SAVED SUCCESSFULLY!")
print("="*60)
print("Model location: models/emotion_model_final.keras")
print("\nYou can now run real_time_detection.py for webcam demo!")