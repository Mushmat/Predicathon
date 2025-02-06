import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define and register your custom learning rate schedule so that load_model can deserialize it.
@tf.keras.utils.register_keras_serializable()
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, decay_steps, alpha=1e-6):
        super(WarmUpCosineDecay, self).__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, step):
        # Warm-up phase
        warmup_lr = self.base_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        # Cosine decay phase
        completed_steps = tf.maximum(tf.cast(step - self.warmup_steps, tf.float32), 0)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * completed_steps / self.decay_steps))
        decayed_lr = (self.base_lr - self.alpha) * cosine_decay + self.alpha
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decayed_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
        }

print("Running Evaluation...")

# Set the path to your test images folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data", "test"))
TEST_IMAGES_PATH = os.path.join(DATA_DIR, "test")

def load_test_images(folder_path, size=(32, 32)):
    """
    Loads and preprocesses all test images from the specified folder.
    Each image is resized to `size` and normalized to [0, 1].
    Returns:
      - A numpy array of preprocessed images.
      - A list of the corresponding image filenames.
    """
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist.")
        return np.array([]), []
    
    images = []
    image_names = []
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if the image cannot be read
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize and normalize
            img = cv2.resize(img, size)
            img = img.astype("float32") / 255.0
            images.append(img)
            image_names.append(img_file)
    return np.array(images), image_names

# Load test images
test_images, image_names = load_test_images(TEST_IMAGES_PATH, size=(32, 32))
print(f"Number of test images loaded: {len(test_images)}")

if len(test_images) > 0:
    # Load the best model from Fold 2 with custom_objects so that WarmUpCosineDecay is recognized.
    model_path = os.getenv("MODEL_PATH", "best_model_fold_2.keras")
    model = load_model(model_path, custom_objects={"WarmUpCosineDecay": WarmUpCosineDecay})
    print("Model loaded successfully!")

    # Make predictions on the test images
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    def get_predictions_and_names():
        """
        Returns:
          - predicted_labels: a numpy array of predicted class indices.
          - image_names: a list of the corresponding image filenames.
        """
        return predicted_labels, image_names

    if __name__ == "__main__":
        # Print out predictions for each test image
        for name, label in zip(image_names, predicted_labels):
            print(f"Image: {name}, Predicted Class: {label}")
else:
    print("No test images found. Exiting.")