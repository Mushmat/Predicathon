import os

# --------------------------
# CPU-Only Settings
# --------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU use
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers

print("✅ Running on CPU.")

# --------------------------
# Hyperparameters & Paths
# --------------------------
BATCH_SIZE = 32
IMG_SIZE = (32, 32)
EPOCHS = 50          # Total number of epochs
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5   # Dropout rate
L2_REG = 1e-4        # L2 regularization factor

# Set dataset directory dynamically using environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data", "train"))

# --------------------------
# Data Pipeline with Augmentation
# --------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.2)
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

# Apply data augmentation to the training dataset only.
def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label

# Scale images to [0, 1]
def scale_input(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Map augmentation and scaling
train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(scale_input, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(scale_input, num_parallel_calls=tf.data.AUTOTUNE)

# Cache and prefetch for improved performance
train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)

# --------------------------
# Learning Rate Scheduler (Cosine Decay)
# --------------------------
cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=1000,
    alpha=1e-6
)

# --------------------------
# Build a Custom CNN Model for 32x32 Images with L2 Regularization
# --------------------------
def build_custom_cnn(input_shape=(32, 32, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_custom_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=2)
model.summary()

# --------------------------
# Compile the Model
# --------------------------
optimizer = optimizers.Adam(learning_rate=cosine_decay)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# --------------------------
# Callbacks: Model Checkpoint & Early Stopping
# --------------------------
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath="best_custom_cnn_improved.keras",
    save_best_only=True,
    monitor="val_accuracy",
    save_weights_only=False,
    verbose=1
)

early_stop_cb = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# --------------------------
# Train the Model
# --------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stop_cb]
)

# --------------------------
# Save the Final Model
# --------------------------
model.save("final_custom_cnn_improved.keras")
