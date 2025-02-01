import os
# --------------------------
# CPU-Only Settings
# --------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU use
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

import tensorflow as tf
print("✅ Running on CPU.")

# --------------------------
# Hyperparameters & Paths
# --------------------------
BATCH_SIZE = 32
IMG_SIZE = (32, 32)
EPOCHS = 50          # Total number of epochs; you can adjust this as needed
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3   # Slight dropout; adjust based on overfitting

DATA_DIR = r"E:/IIITB/Predicathon/project/data/train"

# --------------------------
# Data Pipeline with Augmentation
# --------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),        # increased rotation range
    tf.keras.layers.RandomZoom(0.2),            # increased zoom range
    tf.keras.layers.RandomTranslation(0.1, 0.1),  # add translation
    tf.keras.layers.RandomContrast(0.2)           # increase contrast variation
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

# Apply data augmentation to training dataset only
def augment(image, label):
    image = data_augmentation(image)
    return image, label

# For our custom CNN, we simply scale pixels to [0, 1]
def scale_input(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(scale_input, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(scale_input, num_parallel_calls=tf.data.AUTOTUNE)

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
# Build a Custom CNN Model for 32x32 Images
# --------------------------
from tensorflow.keras import layers, models, optimizers, callbacks

def build_custom_cnn(input_shape=(32, 32, 3), num_classes=2):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(DROPOUT_RATE))
    
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(DROPOUT_RATE))
    
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(DROPOUT_RATE))
    model.add(layers.Dense(num_classes, activation='softmax', dtype='float32'))
    return model

model = build_custom_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=2)
model.summary()

optimizer = optimizers.Adam(learning_rate=cosine_decay)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# --------------------------
# Callbacks: Model Checkpoint & Early Stopping
# --------------------------
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath="best_custom_cnn.keras",  # Save in native Keras format
    save_best_only=True,
    monitor="val_accuracy",
    save_weights_only=False
)

early_stop_cb = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
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
model.save("final_custom_cnn.keras")
