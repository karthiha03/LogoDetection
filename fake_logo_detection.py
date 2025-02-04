import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Load your CSV file
csv_file = "file_mapping.csv"  # Path to your CSV file
df = pd.read_csv(csv_file)

# the first column contains the full paths to the images
image_paths = df.iloc[:, 0].values  # First column as image paths
labels = df['Label'].apply(lambda x: 1 if x == "Genuine" else 0).values  # Second column as labels

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Image Preprocessing Function
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))  # Resize images to 224x224
        img_array = img_to_array(img)  # Convert image to array
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((224, 224, 3))  # Return a blank image as a fallback

# Preprocess the images
X_train_images = np.array([preprocess_image(img_path) for img_path in X_train])
X_val_images = np.array([preprocess_image(img_path) for img_path in X_val])

# Convert labels to numpy arrays
y_train = np.array(y_train)
y_val = np.array(y_val)

# Define the CNN model using Keras
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),  # Input layer with image shape
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_images, y_train, epochs=10, batch_size=32, validation_data=(X_val_images, y_val))

# Save the model
model.save("fake_logo_detection_model.h5")

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val_images, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Function to predict a single image
def predict_image(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Predict using the trained model
    prediction = model.predict(img_array)
    # Interpret the result
    if prediction[0] > 0.5:
        return "Genuine"
    else:
        return "Fake"

while True:
    user_image = input("Enter the path to the .jpg image you want to classify (or type 'exit' to quit): ")
    if user_image.lower() == 'exit':
        print("Exiting the program.")
        break
    elif os.path.exists(user_image) and user_image.lower().endswith('.jpg'):
        result = predict_image(user_image)
        print(f"The image is classified as: {result}")
    else:
        print("Invalid file. Please make sure the file exists and is a .jpg image.")
