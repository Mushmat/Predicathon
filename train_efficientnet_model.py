import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess_data import X_train, X_val, y_train, y_val
from tensorflow.keras.callbacks import EarlyStopping

# Ensure proper EfficientNet Preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input

# Apply Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input  # Ensure EfficientNet pre-processing
)

# Load Pretrained EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Unfreeze last 30 layers to fine-tune
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Add Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)  # Increased neurons
x = Dropout(0.6)(x)  # Higher dropout for better generalization
output = Dense(2, activation='softmax')(x)

# Compile Model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Train Model with Early Stopping
batch_size = 32
epochs = 50

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=1
)

# Save the Model
model.save("efficientnet_deepfake_detector_v2.h5")
print("Model saved as efficientnet_deepfake_detector_v2.h5")
