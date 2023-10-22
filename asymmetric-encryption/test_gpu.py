import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

# Enable memory growth for all GPUs
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# You can verify the setting with the following
for gpu in gpus:
    print(f'{gpu}: Memory growth: {tf.config.experimental.get_memory_growth(gpu)}')
