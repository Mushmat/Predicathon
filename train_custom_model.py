import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from preprocess_data import X_train, X_val, y_train, y_val  # Import preprocessed data

#Step 1: Buidling the custom CNN Model
def build_custom_model(input_shape = (224,224,3)):
    