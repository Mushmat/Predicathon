from load_data import train_images, train_labels
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------------
# Paths & Settings
# --------------------------
train_dir = r"E:/IIITB/Predicathon/project/data/train"
valid_dir = r"E:/IIITB/Predicathon/project/data/validation"
IMG_SIZE = (96, 96)  # Must match the previous settings

# --------------------------
# Preprocess Images: Convert to float32 and apply EfficientNet preprocessing
# --------------------------
def preprocess_images(images):
    images = np.array(images, dtype="float32")
    images = preprocess_input(images)
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
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)
    os.makedirs(os.path.join(valid_dir, "fake"))
    os.makedirs(os.path.join(valid_dir, "real"))

# Save validation images to disk (this is an approximate method to recreate folders)
for i in range(len(X_val)):
    img_name = f"val_{i}.jpg"
    # Determine folder based on label (0: fake, 1: real)
    label_dir = "real" if y_val[i][1] == 1 else "fake"
    img_path = os.path.join(valid_dir, label_dir, img_name)
    # Convert image from RGB to BGR for OpenCV and save
    original_img = X_val[i]
    cv2.imwrite(img_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))

print("Validation images copied successfully.")

# --------------------------
# Save the Preprocessed NumPy Arrays for Training
# --------------------------
np.save(r"E:/IIITB/Predicathon/project/data/X_train.npy", X_train)
np.save(r"E:/IIITB/Predicathon/project/data/y_train.npy", y_train)
np.save(r"E:/IIITB/Predicathon/project/data/X_val.npy", X_val)
np.save(r"E:/IIITB/Predicathon/project/data/y_val.npy", y_val)

print("Preprocessed data saved successfully.")
