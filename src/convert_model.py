import tensorflow as tf

model = tf.keras.models.load_model(
    "../models/efficientnet.keras",
    compile=False
)

model.save(
    "../models/efficientnet.h5"
)

print("Converted successfully")