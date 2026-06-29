from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

from flask import Flask, render_template, request

import tensorflow as tf
import numpy as np
import os
import cv2


app = Flask(__name__)


# ===============================
# PATH SETTINGS
# ===============================

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


# ===============================
# LOAD MODEL
# ===============================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("MODEL LOADED SUCCESSFULLY")


classes = [
    "Cancer",
    "Normal",
    "Precancer"
]



# ===============================
# UPLOAD FOLDER
# ===============================

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



# ===============================
# IMAGE PREPROCESS
# ===============================

def preprocess_image(path):

    img = cv2.imread(path)


    img = cv2.resize(
        img,
        (300,300)
    )


    img = img.astype(
        "float32"
    )


    img = tf.keras.applications.efficientnet.preprocess_input(
        img
    )


    img = np.expand_dims(
        img,
        axis=0
    )


    return img




# ===============================
# GRAD CAM
# ===============================

def generate_gradcam(
        img,
        class_id
):

    try:

        gradcam = Gradcam(
            model,
            model_modifier=ReplaceToLinear()
        )


        score = CategoricalScore(
            class_id
        )


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


        overlay = cv2.addWeighted(
            original,
            0.6,
            colored,
            0.4,
            0
        )


        save_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            "gradcam.jpg"
        )


        cv2.imwrite(
            save_path,
            overlay
        )


        print(
            "GRADCAM SAVED"
        )


        return "gradcam.jpg"


    except Exception as e:

        print(
            "GRADCAM ERROR:",
            e
        )

        return None





# ===============================
# HOME
# ===============================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )





# ===============================
# PREDICTION
# ===============================


@app.route(
    "/predict",
    methods=["POST"]
)

def predict():


    file = request.files["image"]



    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )


    file.save(
        image_path
    )


    print(
        "IMAGE SAVED"
    )



    img = preprocess_image(
        image_path
    )


    print(
        "IMAGE PREPROCESSED"
    )



    prediction = model.predict(
        img
    )


    index = np.argmax(
        prediction[0]
    )


    result = classes[index]


    confidence = float(
        np.max(prediction[0])*100
    )



    print(
        "PREDICTION:",
        result
    )



    # Generate GradCAM always
    gradcam_image = generate_gradcam(
        img,
        index
    )



    return render_template(
        "index.html",
        result=result,
        confidence=round(confidence,2),
        image=file.filename,
        gradcam=gradcam_image
    )





# ===============================
# RUN
# ===============================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=7860
    )