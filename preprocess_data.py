from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np


# Preprocessing (Resize and Normalize)
def preprocess_images(images, size=(224,224)):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img,size) / 255.0
        resized_images.append(resized_img)
    return np.array(resized_images)

