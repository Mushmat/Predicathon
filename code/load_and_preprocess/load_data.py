import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Paths & Settings
# --------------------------
fake_images_path = r"E:/IIITB/Predicathon/project/data/train/fake_cifake_images"
real_images_path = r"E:/IIITB/Predicathon/project/data/train/real_cifake_images"
IMG_SIZE = (32, 32)  # Use 32x32 as provided by the evaluator

# --------------------------
# Function to Load and Resize Images
# --------------------------
def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip unreadable images
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(label)  # For example, 0 for fake, 1 for real
    return np.array(images), np.array(labels)

# --------------------------
# Load Fake and Real Images
# --------------------------
fake_images, fake_labels = load_images_from_folder(fake_images_path, label=0)
real_images, real_labels = load_images_from_folder(real_images_path, label=1)

# Combine datasets
train_images = np.concatenate((fake_images, real_images), axis=0)
train_labels = np.concatenate((fake_labels, real_labels), axis=0)

# Shuffle the dataset to mix classes
shuffled_indices = np.random.permutation(len(train_images))
train_images = train_images[shuffled_indices]
train_labels = train_labels[shuffled_indices]

# --------------------------
# Visualize Data Distribution
# --------------------------
def visualize_data_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(["Fake", "Real"], counts, color=["red", "green"])
    plt.title("Data Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.show()

visualize_data_distribution(train_labels)
print(f"Successfully loaded {len(train_images)} images.")

# --------------------------
# Save Preprocessed Images as NumPy Arrays
# --------------------------
np.save(r"E:/IIITB/Predicathon/project/data/train_images.npy", train_images)
np.save(r"E:/IIITB/Predicathon/project/data/train_labels.npy", train_labels)

print("Data saved successfully.")
