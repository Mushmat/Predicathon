import os
# --------------------------
# CPU-Only Settings
# --------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU use
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

print("✅ Running on CPU.")

# --------------------------
# Hyperparameters & Paths
# --------------------------
BATCH_SIZE = 32
IMG_SIZE = (96, 96)  # Ensure this matches your data preprocessing (or update if needed)
EPOCHS_FROZEN = 5    # Initial training with the base model frozen
EPOCHS_UNFROZEN = 20 # Fine-tuning with the base model unfrozen
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.6

# Directory structure:
# DATA_DIR must have two subfolders: "fake_cifake_images" and "real_cifake_images"
DATA_DIR = r"E:/IIITB/Predicathon/project/data/train"

# --------------------------
# Data Pipeline with Augmentation
# --------------------------
# Create a data augmentation pipeline. You can adjust or add more transforms as needed.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

# Create training and validation datasets using image_dataset_from_directory.
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="training",
    seed=42,
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="validation",
    seed=42,
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Apply EfficientNet preprocessing to the images (this matches the pre-trained model expectations)
def apply_preprocessing(image, label):
    image = preprocess_input(image)
    return image, label

# Chain preprocessing and augmentation for the training dataset
def augment(image, label):
    image = data_augmentation(image)
    return image, label

train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(apply_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(apply_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

# Cache and prefetch for better performance
train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
val_dataset   = val_dataset.cache().prefetch(tf.data.AUTOTUNE)

# --------------------------
# Learning Rate Scheduler (Cosine Decay Restarts)
# --------------------------
cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=LEARNING_RATE,
    first_decay_steps=10,
    t_mul=2,
    m_mul=0.9,
    alpha=1e-6
)

# --------------------------
# Build the Model (EfficientNetB3)
# --------------------------
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models, optimizers, callbacks

# Load EfficientNetB3 with pre-trained ImageNet weights (excluding top layers)
base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False  # Freeze base model initially

# Build the custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='swish'),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(256, activation='swish'),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(2, activation='softmax', dtype='float32')  # Final layer in float32
])

# --------------------------
# Compile the Model
# --------------------------
optimizer = optimizers.AdamW(learning_rate=cosine_decay, weight_decay=1e-5)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------
# Callbacks: Model Checkpoint & Early Stopping
# --------------------------
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath="best_deepfake_model.keras",  # Must end with .keras in TF 2.13+
    save_best_only=True,
    monitor="val_accuracy",
    save_weights_only=False
)

early_stop_cb = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# --------------------------
# Training Phase 1: Frozen Base
# --------------------------
history_frozen = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_FROZEN,
    callbacks=[checkpoint_cb, early_stop_cb]
)

# --------------------------
# Training Phase 2: Fine-Tuning (Unfreeze Base Model)
# --------------------------
base_model.trainable = True

history_unfrozen = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_UNFROZEN,
    callbacks=[checkpoint_cb, early_stop_cb]
)

# --------------------------
# Save Final Model (TF SavedModel Format)
# --------------------------
model.save("efficientnet_deepfake_detector_v2_tf")
