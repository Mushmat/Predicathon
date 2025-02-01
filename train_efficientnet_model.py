import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'  # Enable XLA on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # (Optional) Force CPU only if GPU is causing issues

import tensorflow as tf

# Enable JIT compilation (XLA) for possible CPU speedups
tf.config.optimizer.set_jit(True)

# Check GPU availability
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models, optimizers, callbacks

# Use mixed precision only if GPU is detected
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("⚠️ No GPU detected, disabling mixed precision")
    tf.keras.mixed_precision.set_global_policy('float32')
else:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --------------------
# Hyperparameters
# --------------------
BATCH_SIZE = 32
IMG_SIZE = (96, 96)       # Reduced from (128, 128) to speed up on CPU
EPOCHS_FROZEN = 5         # Train with frozen base model
EPOCHS_UNFROZEN = 20      # Then unfreeze and train further
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.6

# --------------------
# Data Paths
# --------------------
train_dir = "E:/IIITB/Predicathon/project/data/train"
valid_dir = "E:/IIITB/Predicathon/project/data/validation"

# --------------------
# Datasets (with caching and prefetching)
# --------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
).cache().prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
).cache().prefetch(tf.data.AUTOTUNE)

# --------------------
# Learning Rate Scheduler: Cosine Decay Restarts
# --------------------
cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=LEARNING_RATE,
    first_decay_steps=10,
    t_mul=2,
    m_mul=0.9,
    alpha=1e-6
)

# --------------------
# Model Architecture
# --------------------
base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False  # Freeze base model initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='swish'),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(256, activation='swish'),
    layers.Dropout(DROPOUT_RATE),
    # Use float32 for the final layer if mixed precision is enabled
    layers.Dense(2, activation="softmax", dtype="float32")
])

# --------------------
# Compile Model
# --------------------
optimizer = optimizers.AdamW(learning_rate=cosine_decay, weight_decay=1e-5)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------
# Callbacks
# --------------------
checkpoint_cb = callbacks.ModelCheckpoint(
    "efficientnet_deepfake_detector_v2_tf",  # SavedModel format folder
    save_best_only=True,
    monitor="val_accuracy",
    save_format="tf"
)

early_stop_cb = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# --------------------
# 1) Train with Frozen Base Model
# --------------------
history_frozen = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS_FROZEN,
    callbacks=[checkpoint_cb, early_stop_cb]
)

# --------------------
# 2) Unfreeze Base Model for Fine-Tuning
# --------------------
base_model.trainable = True

history_unfrozen = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS_UNFROZEN,  # 20 more epochs
    callbacks=[checkpoint_cb, early_stop_cb]
)

# --------------------
# Save Final Model (TensorFlow SavedModel format)
# --------------------
model.save("efficientnet_deepfake_detector_v2_tf", save_format="tf")
