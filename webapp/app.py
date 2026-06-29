import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2


app = Flask(__name__)


MODEL_PATH="../models/efficientnet.keras"

model=tf.keras.models.load_model(
    MODEL_PATH
)


classes=[
    "Cancer",
    "Normal",
    "Precancer"
]


UPLOAD_FOLDER="static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER



def preprocess_image(path):

    img=cv2.imread(path)

    img=cv2.resize(
        img,
        (300,300)
    )

    img=img.astype("float32")


    img=tf.keras.applications.efficientnet.preprocess_input(
        img
    )


    img=np.expand_dims(
        img,
        axis=0
    )


    return img
def generate_gradcam(img, class_id):

    gradcam = Gradcam(
        model,
        model_modifier=ReplaceToLinear()
    )


    score = CategoricalScore(class_id)


    cam = gradcam(
        score,
        img,
        penultimate_layer=-1
    )


    heatmap = cam[0]


    heatmap = cv2.resize(
        heatmap,
        (300,300)
    )


    heatmap = np.uint8(
        255 * heatmap
    )


    colored = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    original = img[0]


    original = (
        original - original.min()
    ) / (
        original.max()-original.min()
    )


    original = np.uint8(
        original*255
    )


    overlay=cv2.addWeighted(
        original,
        0.6,
        colored,
        0.4,
        0
    )


    path="static/uploads/gradcam.jpg"


    cv2.imwrite(
        path,
        overlay
    )


    return "gradcam.jpg"


@app.route("/")
def home():

    return render_template(
        "index.html"
    )



@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    file=request.files["image"]


    path=os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )


    file.save(path)



    img=preprocess_image(path)



    pred=model.predict(img)


    index=np.argmax(pred[0])
    gradcam_image = generate_gradcam(
    img,
    index
)

    result=classes[index]


    confidence=float(
        np.max(pred[0])*100
    )


    return render_template(
        "index.html",
        result=result,
        confidence=round(confidence,2),
        image=file.filename,
        gradcam=gradcam_image
    )



if __name__=="__main__":

    app.run(
        debug=True
    )