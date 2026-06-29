import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30


train_path = "../data/split_dataset/train"
val_path = "../data/split_dataset/val"


# Augmentation only for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


val_datagen = ImageDataGenerator(
    rescale=1./255
)


train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)


val_data = val_datagen.flow_from_directory(
    val_path,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)



model = Sequential()


model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(IMG_SIZE,IMG_SIZE,3)
    )
)

model.add(MaxPooling2D())


model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D())


model.add(
    Conv2D(
        128,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D())


model.add(Flatten())


model.add(
    Dense(
        256,
        activation="relu"
    )
)

model.add(
    Dropout(0.5)
)


model.add(
    Dense(
        3,
        activation="softmax"
    )
)



model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()



os.makedirs(
    "../models",
    exist_ok=True
)


callbacks = [

EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
),

ModelCheckpoint(
    "../models/baseline_cnn.keras",
    monitor="val_accuracy",
    save_best_only=True
)

]



history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)


print("Training completed")