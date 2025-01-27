import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths
test_images_path = r"E:/IIITB/Predicathon/project/data/test/test"

# Load test images
def load_test_images(folder_path, size=(32, 32)):
    images = []
    image_names = []
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img, size) / 255.0  # Resize and normalize
            images.append(resized_img)
            image_names.append(img_file)  # Keep track of file names for JSON output
    return np.array(images), image_names

# Preprocess test images
test_images, image_names = load_test_images(test_images_path)

# Load the trained model
model = load_model("custom_deepfake_detector_32x32.h5")
print("Model loaded successfully!")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class labels

# Export variables for `generate_predictions.py`
def get_predictions_and_names():
    return predicted_labels, image_names
