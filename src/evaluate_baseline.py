import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = tf.keras.models.load_model(
    "../models/baseline_cnn.keras"
)


test_path="../data/split_dataset/test"


test_gen = ImageDataGenerator(
    rescale=1./255
)


test_data = test_gen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)


pred = model.predict(test_data)


y_pred=np.argmax(pred,axis=1)

y_true=test_data.classes


print("\nCLASSIFICATION REPORT\n")

print(
classification_report(
    y_true,
    y_pred,
    target_names=test_data.class_indices.keys()
)
)


print("\nCONFUSION MATRIX\n")

print(
confusion_matrix(
    y_true,
    y_pred
)
)