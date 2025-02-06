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
DROPOUT_RATE = 0.4    
L2_REG = 1e-4         

# Learning rate schedule hyperparameters
BASE_LR = 0.001
WARMUP_STEPS = 500    
DECAY_STEPS = 2000    
MIN_LR = 1e-6

# Set dataset directories using environment variables or default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data", "train"))

FAKE_DIR = os.path.join(DATA_DIR, "fake_cifake_images")
REAL_DIR = os.path.join(DATA_DIR, "real_cifake_images")

# --------------------------
# Prepare File Paths and Labels
# --------------------------
fake_files = glob.glob(os.path.join(FAKE_DIR, "*"))
real_files = glob.glob(os.path.join(REAL_DIR, "*"))
file_paths = np.array(fake_files + real_files)
labels = np.array([0] * len(fake_files) + [1] * len(real_files))

print(f"Total files found: {len(file_paths)} (Fake: {len(fake_files)}, Real: {len(real_files)})")

# --------------------------
# Albumentations Augmentation Pipeline
# --------------------------
aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomResizedCrop(height=IMG_SIZE[0], width=IMG_SIZE[1], scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5)
])

# --------------------------
# Data Loading Functions
# --------------------------
def load_and_preprocess_image(path, label, training):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = img.numpy()

    if training:
        img_uint8 = (img * 255).astype(np.uint8)
        augmented = aug_pipeline(image=img_uint8)
        img = augmented["image"].astype(np.float32) / 255.0

    label = np.int32(label)
    return img, label

def tf_load_and_preprocess(path, label, training):
    img, lbl = tf.py_function(
        func=load_and_preprocess_image,
        inp=[path, label, training],
        Tout=[tf.float32, tf.int32]
    )
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    lbl.set_shape(())
    lbl = tf.one_hot(lbl, depth=NUM_CLASSES)
    return img, lbl

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
# Custom Learning Rate Schedule
# --------------------------
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, decay_steps, alpha=MIN_LR):
        super(WarmUpCosineDecay, self).__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, step):
        warmup_lr = self.base_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
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

# --------------------------
# K-Fold Cross Validation Training
# --------------------------
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_no = 1
val_accuracies = []

for train_index, val_index in kf.split(file_paths):
    print(f"\n--- Starting Fold {fold_no} ---")
    train_ds = create_dataset(file_paths[train_index], labels[train_index], training=True)
    val_ds = create_dataset(file_paths[val_index], labels[val_index], training=False)

    model = build_custom_cnn()
    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule), 
                  loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Fold {fold_no} - Validation Accuracy: {val_acc:.4f}")
    val_accuracies.append(val_acc)
    fold_no += 1

print(f"\nAverage Validation Accuracy: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
