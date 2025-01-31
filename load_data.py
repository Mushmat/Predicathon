import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Paths
fake_images_path = "E:/IIITB/Predicathon/project/data/train/fake_cifake_images"
real_images_path = "E:/IIITB/Predicathon/project/data/train/real_cifake_images"
test_images_path = "E:/IIITB/Predicathon/project/data/test/test"

# Image size (updated to match EfficientNetB3)
IMG_SIZE = (128, 128)  # Increased from 32x32 to 128x128

# Function to load and resize images
def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)  # Load image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, IMG_SIZE)  # Resize to 128x128
            images.append(img)
            labels.append(label)  # 1 for real, 0 for fake
    return np.array(images), np.array(labels)  # Convert directly to NumPy arrays

# Load training images
fake_images, fake_labels = load_images_from_folder(fake_images_path, label=0)  # 0 for fake
real_images, real_labels = load_images_from_folder(real_images_path, label=1)  # 1 for real

# Combine fake and real data
train_images = np.concatenate((fake_images, real_images), axis=0)
train_labels = np.concatenate((fake_labels, real_labels), axis=0)

# Shuffle data to ensure randomness
shuffled_indices = np.random.permutation(len(train_images))
train_images = train_images[shuffled_indices]
train_labels = train_labels[shuffled_indices]

# Function to visualize dataset distribution
def visualize_data_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(["Fake", "Real"], counts)
    plt.title("Data Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.show()

# Show dataset distribution
visualize_data_distribution(train_labels)
print(f"Successfully loaded {len(train_images)} images.")

# Save preprocessed images for faster training
np.save("E:/IIITB/Predicathon/project/data/train_images.npy", train_images)
np.save("E:/IIITB/Predicathon/project/data/train_labels.npy", train_labels)

print("Data saved successfully.")
