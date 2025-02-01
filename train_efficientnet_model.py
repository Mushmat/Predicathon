import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------------------------------------------
# 1) Optional CPU/GPU Settings
# ----------------------------------------------------------------
# If GPU is giving you trouble, uncomment the following line to force CPU use:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Enable XLA on CPU (or GPU)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

import tensorflow as tf

# Enable JIT compilation (XLA) in TensorFlow for potential speed gains
tf.config.optimizer.set_jit(True)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available:", gpus)

# If you want to use mixed precision only when a GPU is present:
if len(gpus) == 0:
    print("⚠️ No GPU detected; using float32.")
    tf.keras.mixed_precision.set_global_policy('float32')
else:
    print("✅ GPU detected; enabling mixed precision.")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ----------------------------------------------------------------
# 2) Hyperparameters
# ----------------------------------------------------------------
BATCH_SIZE = 32
IMG_SIZE = (96, 96)       # Reduced size for faster CPU training
EPOCHS_FROZEN = 5         # Train with the base model frozen
EPOCHS_UNFROZEN = 20      # Then unfreeze for fine-tuning
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.6

# Directory: Must contain subfolders "fake_cifake_images" and "real_cifake_images"
DATA_DIR = r"E:/IIITB/Predicathon/project/data/train"

# ----------------------------------------------------------------
# 3) Build Train/Val Datasets Directly from Directory
# ----------------------------------------------------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",              # Infer labels from subfolder names
    label_mode="categorical",       # 2D one-hot labels
    validation_split=0.2,           # 20% for validation
    subset="training",              # This dataset is the "training" subset
    seed=42,                        # For reproducible splits
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="validation",            # This dataset is the "validation" subset
    seed=42,
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Optional: Prefetch for performance
train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
val_dataset   = val_dataset.cache().prefetch(tf.data.AUTOTUNE)

# ----------------------------------------------------------------
# 4) Learning Rate Scheduler (Cosine Decay Restarts)
# ----------------------------------------------------------------
cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=LEARNING_RATE,
    first_decay_steps=10,
    t_mul=2,
    m_mul=0.9,
    alpha=1e-6
)

# ----------------------------------------------------------------
# 5) Build the Model (EfficientNetB3)
# ----------------------------------------------------------------
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models, optimizers, callbacks

base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze the base model initially (transfer learning)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='swish'),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(256, activation='swish'),
    layers.Dropout(DROPOUT_RATE),
    # Use float32 for the final layer if using mixed precision
    layers.Dense(2, activation='softmax', dtype='float32')
])

# ----------------------------------------------------------------
# 6) Compile the Model
# ----------------------------------------------------------------
optimizer = optimizers.AdamW(learning_rate=cosine_decay, weight_decay=1e-5)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------------------------------------
# 7) Define Callbacks (Checkpoints, Early Stopping)
# ----------------------------------------------------------------
# NOTE: The `.keras` extension is now required if saving the full model
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath="best_deepfake_model.keras",  # must end with .keras in TF 2.13+
    save_best_only=True,
    monitor="val_accuracy",
    save_weights_only=False  # saving the full model in .keras format
)

early_stop_cb = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# ----------------------------------------------------------------
# 8) Training - Part 1 (Frozen Base)
# ----------------------------------------------------------------
history_frozen = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_FROZEN,   # 5 epochs
    callbacks=[checkpoint_cb, early_stop_cb]
)

# ----------------------------------------------------------------
# 9) Unfreeze & Fine-tune
# ----------------------------------------------------------------
base_model.trainable = True

history_unfrozen = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_UNFROZEN,  # 20 epochs
    callbacks=[checkpoint_cb, early_stop_cb]
)

# ----------------------------------------------------------------
# 10) Save Final Model in TF SavedModel Format (Optional)
# ----------------------------------------------------------------
# If you also want to save in the older "SavedModel" folder format:
model.save("efficientnet_deepfake_detector_v2_tf")  
