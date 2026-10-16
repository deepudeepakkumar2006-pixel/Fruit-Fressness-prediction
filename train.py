import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os

# ==============================
# 1. DATASET PATHS
# ==============================

train_dir = r"D:\pbl food freshness detection dataset\train"
valid_dir = r"D:\pbl food freshness detection dataset\valid"
test_dir = r"D:\pbl food freshness detection dataset\test"

# ==============================
# 2. BASIC SETTINGS
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4

# ==============================
# 3. LOAD DATASET
# ==============================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Display class names
class_names = train_dataset.class_names

print("\nClass names:")
print(class_names)

# ==============================
# 4. DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# ==============================
# 5. PRETRAINED MODEL
# ==============================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# ==============================
# 6. BUILD MODEL
# ==============================

model = models.Sequential([
    data_augmentation,

    layers.Rescaling(1./127.5, offset=-1),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.2),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

# ==============================
# 7. COMPILE MODEL
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# 8. DISPLAY MODEL
# ==============================

model.summary()

# ==============================
# 9. TRAIN MODEL
# ==============================

EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

# ==============================
# 10. TEST MODEL
# ==============================

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# ==============================
# 11. SAVE MODEL
# ==============================

os.makedirs("models", exist_ok=True)

model.save("models/banana_freshness_model.keras")

print("\nModel saved successfully!")
print("Location: models/banana_freshness_model.keras")