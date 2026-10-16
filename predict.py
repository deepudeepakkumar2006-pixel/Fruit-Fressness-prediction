import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = r"D:\pbl food freshness project\models\food_freshness_18class_model.keras"

print("Loading multi-fruit freshness model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ============================================================
# CLASS NAMES
# IMPORTANT: Alphabetical order used by image_dataset_from_directory
# ============================================================

CLASS_NAMES = [
    "apple_fresh",
    "apple_rotten",
    "grape_fresh",
    "grape_rotten",
    "guava_fresh",
    "guava_rotten",
    "jujube_fresh",
    "jujube_rotten",
    "orange_fresh",
    "orange_rotten",
    "overripe",
    "pomegranate_fresh",
    "pomegranate_rotten",
    "ripe",
    "rotten",
    "strawberry_fresh",
    "strawberry_rotten",
    "unripe"
]

# ============================================================
# IMAGE TO TEST
# ============================================================

IMAGE_PATH = r"D:\pbl food freshness multi dataset\test\apple_fresh\apple_fresh_00072.jpg"

# ============================================================
# LOAD IMAGE
# ============================================================

image = load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

image_array = img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)

# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(predictions[0])

predicted_class = CLASS_NAMES[predicted_index]

confidence = float(
    predictions[0][predicted_index] * 100
)

# ============================================================
# FORMAT RESULT
# ============================================================

if "_" in predicted_class:

    fruit, freshness = predicted_class.split("_", 1)

    fruit = fruit.capitalize()
    freshness = freshness.capitalize()

else:

    # These are banana-only classes:
    # ripe, unripe, overripe, rotten

    fruit = "Banana"
    freshness = predicted_class.capitalize()

# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 50)
print("FOOD FRESHNESS DETECTION RESULT")
print("=" * 50)

print(f"Fruit      : {fruit}")
print(f"Freshness  : {freshness}")
print(f"Confidence : {confidence:.2f}%")

print("=" * 50)