import tensorflow as tf
import numpy as np

from sklearn.metrics import (
classification_report,
confusion_matrix
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = tf.keras.models.load_model(
    "../models/efficientnet.keras"
)


test_path="../data/split_dataset/test"


test_gen=ImageDataGenerator(
    preprocessing_function=
    tf.keras.applications.efficientnet.preprocess_input
)


test=test_gen.flow_from_directory(
    test_path,
    target_size=(300,300),
    batch_size=8,
    class_mode="categorical",
    shuffle=False
)


pred=model.predict(test)


y_pred=np.argmax(
    pred,
    axis=1
)


y_true=test.classes


print("\nCLASSIFICATION REPORT\n")


print(
classification_report(
    y_true,
    y_pred,
    target_names=test.class_indices.keys()
)
)


print("\nCONFUSION MATRIX\n")


print(
confusion_matrix(
    y_true,
    y_pred
)
)