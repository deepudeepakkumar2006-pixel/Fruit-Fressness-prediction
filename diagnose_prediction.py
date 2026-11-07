
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

fruit_model = tf.keras.models.load_model(
    os.path.join(PROJECT_DIR, "models", "food_freshness_18class_model.keras")
)

vegetable_model = tf.keras.models.load_model(
    os.path.join(PROJECT_DIR, "models", "vegetable_freshness_model.keras")
)

FRUIT_CLASSES = [
    "apple_fresh", "apple_rotten",
    "grape_fresh", "grape_rotten",
    "guava_fresh", "guava_rotten",
    "jujube_fresh", "jujube_rotten",
    "orange_fresh", "orange_rotten",
    "overripe",
    "pomegranate_fresh", "pomegranate_rotten",
    "ripe", "rotten",
    "strawberry_fresh", "strawberry_rotten",
    "unripe"
]

VEGETABLE_CLASSES = [
    "Bellpepper__Fresh", "Bellpepper__Rotten",
    "Carrot__Fresh", "Carrot__Rotten",
    "Cucumber__Fresh", "Cucumber__Rotten",
    "Potato__Fresh", "Potato__Rotten",
    "Tomato__Fresh", "Tomato__Rotten"
]

image_path = input("Enter the full path of your grape image: ").strip().strip('"')

image = load_img(image_path, target_size=(224, 224))
image_array = np.expand_dims(img_to_array(image), axis=0)

def show_top_predictions(model, classes, title):
    predictions = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:3]

    print(f"\n{title}")
    for index in top_indices:
        print(f"{classes[index]}: {predictions[index] * 100:.2f}%")

show_top_predictions(fruit_model, FRUIT_CLASSES, "FRUIT MODEL")
show_top_predictions(vegetable_model, VEGETABLE_CLASSES, "VEGETABLE MODEL")
