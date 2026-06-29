from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

from flask import Flask, render_template, request

import tensorflow as tf
import numpy as np
import os
import cv2


app = Flask(__name__)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnet.h5"
)


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("MODEL LOADED")


classes = [
    "Cancer",
    "Normal",
    "Precancer"
]


UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "static",
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



def preprocess_image(path):

    img = cv2.imread(path)

    img = cv2.resize(
        img,
        (300,300)
    )

    img = img.astype("float32")


    img = tf.keras.applications.efficientnet.preprocess_input(
        img
    )


    img = np.expand_dims(
        img,
        axis=0
    )

    return img




def generate_gradcam(img, class_id):

    try:

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


        original = img[0]

        original = (
            original-original.min()
        ) / (
            original.max()-original.min()
        )

        original = np.uint8(
            original*255
        )


        colored = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )


        overlay = cv2.addWeighted(
            original,
            0.7,
            colored,
            0.3,
            0
        )


        save_path=os.path.join(
            app.config["UPLOAD_FOLDER"],
            "gradcam.jpg"
        )


        cv2.imwrite(
            save_path,
            overlay
        )


        print("GRADCAM SAVED")

        return "gradcam.jpg"


    except Exception as e:

        print("GRADCAM ERROR:",e)

        return None




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

    file = request.files["image"]


    path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )


    file.save(path)


    print("IMAGE SAVED")


    img = preprocess_image(path)


    print("PREPROCESS DONE")


    pred = model.predict(img)


    index = np.argmax(
        pred[0]
    )


    print(
        "PREDICTION:",
        classes[index]
    )


    gradcam_image = None

if index in [0,2]:
    gradcam_image = generate_gradcam(
        img,
        index
    )


    confidence = float(
        np.max(pred[0])*100
    )


    return render_template(
        "index.html",
        result=classes[index],
        confidence=round(confidence,2),
        image=file.filename,
        gradcam=gradcam_image
    )




if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )