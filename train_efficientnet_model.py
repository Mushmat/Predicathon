import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import os

# Ensure mixed precision training for efficiency
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Hyperparameters
BATCH_SIZE = 32
IMG_SIZE = (128, 128)  # Upgraded from 32x32 to 128x128 for better feature extraction
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.6

# Augmentation strategy
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Data Loading
train_dir = "E:/IIITB/Predicathon/project/data/train"
valid_dir = "E:/IIITB/Predicathon/project/data/validation"

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model Architecture
base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = True  # Fine-tuning entire model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='swish'),  # Swish activation instead of ReLU
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(256, activation='swish'),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(2, activation="softmax", dtype="float32")
])

# Compile Model
optimizer = optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-5)  # AdamW improves generalization
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
checkpoint_cb = callbacks.ModelCheckpoint(
    "efficientnet_deepfake_detector_v2.keras", 
    save_best_only=True, 
    monitor="val_accuracy"
)

early_stop_cb = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=10, restore_best_weights=True
)

lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
)

cosine_annealing = callbacks.CosineDecayRestarts(
    initial_learning_rate=LEARNING_RATE, first_decay_steps=10, t_mul=2, m_mul=0.9, alpha=1e-6
)

# Training
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stop_cb, lr_scheduler, cosine_annealing],
    workers=4,  # Enables multi-threaded data loading
    use_multiprocessing=True
)

# Save Model
model.save("efficientnet_deepfake_detector_v2.keras")
