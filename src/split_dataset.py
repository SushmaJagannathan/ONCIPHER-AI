import os
import shutil
import random


source = "../data/dataset"

destination = "../data/split_dataset"


classes = [
    "Normal",
    "Cancer"
]


random.seed(42)


for cls in classes:

    images=[]

    folder=os.path.join(source,cls)

    for img in os.listdir(folder):

        if img.lower().endswith(
            (".jpg",".jpeg",".png")
        ):
            images.append(img)


    random.shuffle(images)


    n=len(images)

    train_end=int(0.7*n)
    val_end=int(0.85*n)


    splits={
        "train":images[:train_end],
        "val":images[train_end:val_end],
        "test":images[val_end:]
    }


    for split,files in splits.items():

        out=os.path.join(
            destination,
            split,
            cls
        )

        os.makedirs(
            out,
            exist_ok=True
        )


        for f in files:

            shutil.copy(
                os.path.join(folder,f),
                os.path.join(out,f)
            )



# Precancer including subfolders

precancer_images=[]

precancer_path=os.path.join(
    source,
    "Precancer"
)


for root,dirs,files in os.walk(precancer_path):

    for f in files:

        if f.lower().endswith(
            (".jpg",".jpeg",".png")
        ):
            precancer_images.append(
                os.path.join(root,f)
            )



random.shuffle(precancer_images)


n=len(precancer_images)

splits={
"train":precancer_images[:int(0.7*n)],
"val":precancer_images[int(0.7*n):int(0.85*n)],
"test":precancer_images[int(0.85*n):]
}



for split,files in splits.items():

    out=os.path.join(
        destination,
        split,
        "Precancer"
    )

    os.makedirs(out,exist_ok=True)


    for f in files:

        shutil.copy(
            f,
            os.path.join(
                out,
                os.path.basename(f)
            )
        )


print("Dataset split completed")