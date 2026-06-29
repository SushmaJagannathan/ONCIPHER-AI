import os
from PIL import Image

dataset_path = "../data/dataset"

valid_extensions = (".jpg", ".jpeg", ".png")

total = 0

for root, folders, files in os.walk(dataset_path):

    images = [
        f for f in files
        if f.lower().endswith(valid_extensions)
    ]

    if images:
        print("\nFolder:", root)
        print("Images:", len(images))

        for img in images[:3]:
            try:
                image_path = os.path.join(root, img)
                im = Image.open(image_path)

                print(
                    img,
                    "Size:",
                    im.size
                )

            except Exception:
                print(
                    "BAD IMAGE:",
                    img
                )

        total += len(images)


print("\n================")
print("TOTAL IMAGES:", total)
print("================")