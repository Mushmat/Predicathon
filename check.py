import tensorflow as tf

print("Is TensorFlow built with CUDA:", tf.test.is_built_with_cuda())

print("Physical GPUs TF sees:", tf.config.list_physical_devices('GPU'))
