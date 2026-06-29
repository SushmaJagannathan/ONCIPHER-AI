import tensorflow as tf
import cv2
import numpy as np
import os

from tensorflow.keras.models import load_model
from PIL import Image

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore



model = load_model(
    "../models/efficientnet.keras"
)


image_path="../data/split_dataset/test/Precancer"


image_name=os.listdir(image_path)[0]


img=cv2.imread(
    os.path.join(
        image_path,
        image_name
    )
)


img=cv2.resize(
    img,
    (300,300)
)


x=np.expand_dims(
    img/255.0,
    axis=0
)



pred=model.predict(x)

class_id=np.argmax(pred[0])


score=CategoricalScore(class_id)


gradcam=Gradcam(
    model,
    model_modifier=ReplaceToLinear()
)


cam=gradcam(
    score,
    x,
    penultimate_layer=-1
)



heatmap=cam[0]


heatmap=cv2.resize(
    heatmap,
    (300,300)
)


heatmap=np.uint8(
    255*heatmap
)


colored=cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


overlay=cv2.addWeighted(
    img,
    0.6,
    colored,
    0.4,
    0
)


os.makedirs(
    "../results/gradcam",
    exist_ok=True
)


cv2.imwrite(
    "../results/gradcam/result.jpg",
    overlay
)


print(
"Saved GradCAM"
)

print(
"Prediction:",
class_id
)