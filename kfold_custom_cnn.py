import os
import glob
import numpy as np
import tensorflow as tf
import albumentations as A
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers

# --------------------------
# Settings & Hyperparameters
# --------------------------
BATCH_SIZE = 32
IMG_SIZE = (32, 32)
NUM_CLASSES = 2
EPOCHS = 50
NUM_FOLDS = 5

# Regularization hyperparameters
DROPOUT_RATE = 0.4    # Slightly lower dropout
L2_REG = 1e-4         # L2 regularization factor

# Learning rate schedule hyperparameters
BASE_LR = 0.001
WARMUP_STEPS = 500    # Number of steps for warm-up
DECAY_STEPS = 2000    # Cosine decay steps
MIN_LR = 1e-6

# Directories for the two classes (adjust these paths as needed)
FAKE_DIR = r"E:/IIITB/Predicathon/project/data/train/fake_cifake_images"
REAL_DIR = r"E:/IIITB/Predicathon/project/data/train/real_cifake_images"

# --------------------------
# Prepare File Paths and Labels
# --------------------------
# Assume that images in FAKE_DIR belong to class 0 and those in REAL_DIR to class 1.
fake_files = glob.glob(os.path.join(FAKE_DIR, "*"))
real_files = glob.glob(os.path.join(REAL_DIR, "*"))
file_paths = np.array(fake_files + real_files)
labels = np.array([0] * len(fake_files) + [1] * len(real_files))

print(f"Total files found: {len(file_paths)} (Fake: {len(fake_files)}, Real: {len(real_files)})")

# --------------------------
# Albumentations Augmentation Pipeline
# --------------------------
# This pipeline will be applied only during training.
aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),  # rotate up to 20 degrees
    A.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5)
])

def load_and_preprocess_image(path, label, training):
    """
    Reads an image from a file, decodes it, resizes it,
    scales it to [0,1], and (if training) applies augmentation.
    """
    # Read file
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = img.numpy()  # convert to numpy array for albumentations
    
    if training:
        # Albumentations expects uint8 images; we convert to uint8 and back.
        img_uint8 = (img * 255).astype(np.uint8)
        augmented = aug_pipeline(image=img_uint8)
        img = augmented["image"].astype(np.float32) / 255.0

    # Ensure the label is int32.
    label = np.int32(label)
    return img, label

def tf_load_and_preprocess(path, label, training):
    # Wrap the Python function using tf.py_function
    img, lbl = tf.py_function(
        func=load_and_preprocess_image,
        inp=[path, label, training],
        Tout=[tf.float32, tf.int32]
    )
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    lbl.set_shape(())
    # One-hot encode label
    lbl = tf.one_hot(lbl, depth=NUM_CLASSES)
    return img, lbl

# --------------------------
# Create a tf.data.Dataset for a given list of file paths and labels
# --------------------------
def create_dataset(file_paths, labels, training):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(lambda path, lbl: tf_load_and_preprocess(path, lbl, training),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
    return dataset

# --------------------------
# Custom Learning Rate Schedule with Warm-Up and Cosine Decay
# --------------------------
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, decay_steps, alpha=MIN_LR):
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

lr_schedule = WarmUpCosineDecay(base_lr=BASE_LR, warmup_steps=WARMUP_STEPS, decay_steps=DECAY_STEPS)

# --------------------------
# Define the Custom CNN Model
# --------------------------
def build_custom_cnn(input_shape=(32, 32, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Block 2
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Block 3
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# --------------------------
# K-Fold Cross Validation Training
# --------------------------
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_no = 1
val_accuracies = []

for train_index, val_index in kf.split(file_paths):
    print(f"\n--- Starting Fold {fold_no} ---")
    train_files = file_paths[train_index]
    train_labels = labels[train_index]
    val_files = file_paths[val_index]
    val_labels = labels[val_index]
    
    # Create tf.data.Datasets for this fold
    train_ds = create_dataset(train_files, train_labels, training=True)
    val_ds = create_dataset(val_files, val_labels, training=False)
    
    # Build and compile model for this fold
    model = build_custom_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES)
    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Define callbacks: ModelCheckpoint and EarlyStopping
    ckpt = callbacks.ModelCheckpoint(
        filepath=f"best_model_fold_{fold_no}.keras",
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ckpt, early_stop],
        verbose=2
    )
    
    # Evaluate on the validation fold
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Fold {fold_no} - Validation Accuracy: {val_acc:.4f}")
    val_accuracies.append(val_acc)
    fold_no += 1

print("\n--- Cross Validation Results ---")
print(f"Average Validation Accuracy: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
