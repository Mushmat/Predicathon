from load_data import train_images, train_labels
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input  # ✅ Add this import

# Ensure images are correctly normalized for EfficientNet
def preprocess_images(images):
    images = np.array(images, dtype="float32")  # Ensure correct data type
    images = preprocess_input(images)  # Apply EfficientNet preprocessing
    return images

# Preprocess images
train_images = preprocess_images(train_images)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Convert labels to categorical format
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
