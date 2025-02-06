import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Paths & Settings
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data", "train"))

fake_images_path = os.path.join(DATA_DIR, "fake_cifake_images")
real_images_path = os.path.join(DATA_DIR, "real_cifake_images")
IMG_SIZE = (32, 32)  # Use 32x32 as provided by the evaluator

# --------------------------
# Function to Load and Resize Images
# --------------------------
def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist.")
        return np.array([]), np.array([])
    
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

if len(fake_images) == 0 or len(real_images) == 0:
    print("Error: One or more image directories are empty or missing.")
    exit()

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
out_dir = os.getenv("OUT_DIR", os.path.join(BASE_DIR, "data"))
np.save(os.path.join(out_dir, "train_images.npy"), train_images)
np.save(os.path.join(out_dir, "train_labels.npy"), train_labels)

print("Data saved successfully.")
