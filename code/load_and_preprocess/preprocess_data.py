import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --------------------------
# Paths & Settings
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data", "train"))
VALID_DIR = os.getenv("VALID_DIR", os.path.join(BASE_DIR, "data", "validation"))

fake_images_path = os.path.join(DATA_DIR, "fake_cifake_images")
real_images_path = os.path.join(DATA_DIR, "real_cifake_images")
IMG_SIZE = (32, 32)  # Must match load_data.py

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
# Preprocess Images: Convert to float32 and scale pixels to [0, 1]
# --------------------------
def preprocess_images(images):
    images = np.array(images, dtype="float32") / 255.0
    return images

train_images = preprocess_images(train_images)

# --------------------------
# Split into Training and Validation Sets
# --------------------------
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

# Convert labels to categorical format (2 classes: fake and real)
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")

# --------------------------
# (Optional) Copy Validation Images to a Folder Structure
# --------------------------
if not os.path.exists(VALID_DIR):
    os.makedirs(VALID_DIR)
    os.makedirs(os.path.join(VALID_DIR, "fake"))
    os.makedirs(os.path.join(VALID_DIR, "real"))

for i in range(len(X_val)):
    img_name = f"val_{i}.jpg"
    label_dir = "real" if y_val[i][1] == 1 else "fake"
    img_path = os.path.join(VALID_DIR, label_dir, img_name)
    original_img = (X_val[i] * 255).astype("uint8")
    cv2.imwrite(img_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))

print("Validation images copied successfully.")

# --------------------------
# Save the Preprocessed NumPy Arrays for Training
# --------------------------
OUT_DIR = os.getenv("OUT_DIR", os.path.join(BASE_DIR, "data"))
np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUT_DIR, "y_val.npy"), y_val)

print("Preprocessed data saved successfully.")