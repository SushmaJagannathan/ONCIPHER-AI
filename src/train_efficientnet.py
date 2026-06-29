import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
EarlyStopping,
ModelCheckpoint
)

import os


IMG=300
BATCH=8
EPOCHS=25


train_path="../data/split_dataset/train"
val_path="../data/split_dataset/val"



train_gen=ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)


val_gen=ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)



train=train_gen.flow_from_directory(
    train_path,
    target_size=(IMG,IMG),
    batch_size=BATCH,
    class_mode="categorical"
)


val=val_gen.flow_from_directory(
    val_path,
    target_size=(IMG,IMG),
    batch_size=BATCH,
    class_mode="categorical"
)



base=EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG,IMG,3)
)


base.trainable=False



x=base.output

x=GlobalAveragePooling2D()(x)

x=Dense(
    256,
    activation="relu"
)(x)

x=Dropout(0.4)(x)


output=Dense(
    3,
    activation="softmax"
)(x)



model=Model(
    base.input,
    output
)


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


os.makedirs(
    "../models",
    exist_ok=True
)


history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    callbacks=[
        EarlyStopping(
        patience=5,
        restore_best_weights=True
        ),

        ModelCheckpoint(
        "../models/efficientnet.keras",
        save_best_only=True
        )
    ]
)


print("EfficientNet training finished")
import pickle

with open("../models/history.pkl","wb") as f:
    pickle.dump(history.history,f)