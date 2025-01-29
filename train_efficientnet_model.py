import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess_data import X_train, X_val, y_train, y_val  # Import preprocessed data
from tensorflow.keras.callbacks import EarlyStopping

# Apply Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Step 1: Load the Pretrained EfficientNet Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False  # Freeze base model to use pretrained weights

# Step 2: Add Custom Layers for Classification
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling layer
x = Dense(128, activation='relu')(x)  # Dense layer with 128 neurons
x = Dropout(0.5)(x)  # Dropout for regularization
output = Dense(2, activation='softmax')(x)  # Final output layer for 2 classes (real/fake)

# Step 3: Compile the Model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Step 4: Train the Model
batch_size = 32
epochs = 50  # Increased max epochs with early stopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=1
)

# Step 5: Save the Model
model.save("efficientnet_deepfake_detector.h5")
print("Model saved as efficientnet_deepfake_detector.h5")
