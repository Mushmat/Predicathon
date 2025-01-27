from load_data import train_images, train_labels

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np


# Preprocessing (Resize and Normalize)
def preprocess_images(images, size=(32,32)):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img,size) / 255.0
        resized_images.append(resized_img)
    return np.array(resized_images)

#Preprocess
train_images = preprocess_images(train_images, size=(32, 32))
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)


#Convert labels to categorical for classification
y_train = to_categorical(y_train, num_classes = 2)
y_val = to_categorical(y_val, num_classes = 2)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")