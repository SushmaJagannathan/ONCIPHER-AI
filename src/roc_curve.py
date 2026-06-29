import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from tensorflow.keras.preprocessing.image import ImageDataGenerator


model=tf.keras.models.load_model(
    "../models/efficientnet.keras"
)


test_gen=ImageDataGenerator(
    preprocessing_function=
    tf.keras.applications.efficientnet.preprocess_input
)


test=test_gen.flow_from_directory(
    "../data/split_dataset/test",
    target_size=(300,300),
    batch_size=8,
    class_mode="categorical",
    shuffle=False
)


y_true=test.classes

y_score=model.predict(test)


y_true_bin=label_binarize(
    y_true,
    classes=[0,1,2]
)


plt.figure(figsize=(7,6))


for i in range(3):

    fpr,tpr,_=roc_curve(
        y_true_bin[:,i],
        y_score[:,i]
    )

    roc_auc=auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"Class {i} AUC={roc_auc:.2f}"
    )


plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)


plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "EfficientNetB3 ROC Curve"
)

plt.legend()


plt.savefig(
    "../results/roc_curve.png",
    dpi=300
)

print("ROC saved")