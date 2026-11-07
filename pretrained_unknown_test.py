
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.utils import load_img, img_to_array

# ------------------------------------------------
# PRETRAINED MODEL
# ------------------------------------------------

print("Loading pretrained ImageNet MobileNetV2...")

recognition_model = MobileNetV2(
    weights="imagenet",
    include_top=True
)

print("Pretrained model loaded successfully!")


# ------------------------------------------------
# IMAGE TESTING
# ------------------------------------------------

def inspect_image(image_path, selected_category):

    if not os.path.isfile(image_path):
        print("\nImage not found:", image_path)
        return

    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    predictions = recognition_model.predict(
        image_array,
        verbose=0
    )

    decoded = decode_predictions(
        predictions,
        top=5
    )[0]

    print("\n" + "=" * 55)
    print("IMAGE:", os.path.basename(image_path))
    print("SELECTED CATEGORY:", selected_category)
    print("\nPRETRAINED MODEL PREDICTIONS:")

    for rank, (class_id, label, confidence) in enumerate(
        decoded, start=1
    ):
        print(
            f"{rank}. {label} - {confidence * 100:.2f}%"
        )


# ------------------------------------------------
# MAIN
# ------------------------------------------------

if __name__ == "__main__":

    print("\nPRETRAINED MODEL DIAGNOSTIC TEST")

    for category in ["FRUITS", "VEGETABLES"]:

        print("\n" + "#" * 55)
        print(category)
        print("#" * 55)

        paths = input(
            "Enter image paths separated by semicolons: "
        ).strip()

        if not paths:
            continue

        for path in paths.split(";"):
            inspect_image(path.strip(), category)

    print("\nTesting completed.")
