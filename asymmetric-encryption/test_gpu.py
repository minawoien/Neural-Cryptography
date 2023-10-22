import tensorflow as tf
# Enable log device placement
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Configure TensorFlow session
tf.config.set_soft_device_placement(True)

# Check the device being used
with tf.device('GPU:0'):  # Or 'GPU:1' etc. depending on which GPU you want to use
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

# Should print something like '/job:localhost/replica:0/task:0/device:GPU:0'
print(c.device)
