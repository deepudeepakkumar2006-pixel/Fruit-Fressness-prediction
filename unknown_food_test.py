
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import image_dataset_from_directory, load_img, img_to_array

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = r"D:\pbl food freshness project"

CONFIGS = {
    "FRUITS": {
        "model": os.path.join(
            BASE_DIR, "models", "food_freshness_18class_model.keras"
        ),
        "dataset": r"D:\pbl food freshness multi dataset",
    },
    "VEGETABLES": {
        "model": os.path.join(
            BASE_DIR, "models", "vegetable_freshness_model.keras"
        ),
        "dataset": os.path.join(BASE_DIR, "vegetable_split_dataset"),
    },
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
REFERENCE_LIMIT = 100
VALID_PERCENTILE = 5


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(feature_model, dataset):
    all_features = []
    all_labels = []

    for images, labels in dataset:
        features = feature_model.predict(images, verbose=0)

        all_features.append(features)
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_features),
        np.concatenate(all_labels)
    )


def normalize_features(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-10)


# --------------------------------------------------
# BUILD CLASS REFERENCES
# --------------------------------------------------

def prepare_category(category, config):

    print("\n" + "=" * 65)
    print("PROCESSING:", category)
    print("=" * 65)

    train_dir = os.path.join(config["dataset"], "train")
    valid_dir = os.path.join(config["dataset"], "valid")

    if not os.path.isdir(train_dir) or not os.path.isdir(valid_dir):
        raise FileNotFoundError(
            f"Training or validation folder not found for {category}"
        )

    model = load_model(config["model"])

    feature_layer = model.get_layer("global_average_pooling2d")
    feature_model = Model(
        inputs=model.input,
        outputs=feature_layer.output
    )

    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    valid_ds = image_dataset_from_directory(
        valid_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names

    if valid_ds.class_names != class_names:
        raise ValueError("Training and validation class folders do not match.")

    if len(class_names) != model.output_shape[-1]:
        raise ValueError(
            f"Class count mismatch: {len(class_names)} folders, "
            f"{model.output_shape[-1]} model outputs."
        )

    print("\nClasses:", class_names)

    print("\nExtracting training features...")
    train_features, train_labels = extract_features(
        feature_model, train_ds
    )

    print("\nExtracting validation features...")
    valid_features, valid_labels = extract_features(
        feature_model, valid_ds
    )

    train_features = normalize_features(train_features)
    valid_features = normalize_features(valid_features)

    # Build one reference vector (centroid) per class.
    centroids = []

    for class_index, class_name in enumerate(class_names):

        class_features = train_features[
            train_labels == class_index
        ]

        if len(class_features) == 0:
            raise ValueError(f"No training images for {class_name}")

        # Limit the number of reference images per class.
        class_features = class_features[:REFERENCE_LIMIT]

        centroid = np.mean(class_features, axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-10)

        centroids.append(centroid)

    centroids = np.array(centroids)

    # Compare validation images against all class references.
    validation_scores = valid_features @ centroids.T

    # Similarity to the correct class reference.
    correct_scores = validation_scores[
        np.arange(len(valid_labels)), valid_labels
    ]

    # Calibrate threshold using known validation images.
    threshold = float(
        np.percentile(correct_scores, VALID_PERCENTILE)
    )

    # Measure how many known validation images are rejected.
    validation_max_scores = np.max(validation_scores, axis=1)
    rejected_known = np.sum(validation_max_scores < threshold)

    print("\nVALIDATION RESULTS")
    print("Images:", len(valid_labels))
    print("Similarity threshold:", round(threshold, 4))
    print(
        "Known validation images rejected:",
        f"{rejected_known}/{len(valid_labels)}"
    )

    return {
        "model": model,
        "feature_model": feature_model,
        "class_names": class_names,
        "centroids": centroids,
        "threshold": threshold
    }


# --------------------------------------------------
# TEST UNKNOWN IMAGES
# --------------------------------------------------

def test_image(category, image_path, data):

    if not os.path.isfile(image_path):
        print("\nImage not found:", image_path)
        return

    image = load_img(image_path, target_size=IMG_SIZE)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    features = data["feature_model"].predict(
        image_array, verbose=0
    )

    features = normalize_features(features)

    scores = features @ data["centroids"].T

    best_index = int(np.argmax(scores[0]))
    best_score = float(scores[0][best_index])

    predicted_class = data["class_names"][best_index]
    threshold = data["threshold"]

    print("\nImage:", image_path)
    print("Selected category:", category)
    print("Closest class:", predicted_class)
    print("Similarity:", round(best_score, 4))
    print("Threshold:", round(threshold, 4))

    if best_score >= threshold:
        print("RESULT: ACCEPTED - Similar to a trained class")
    else:
        print("RESULT: UNKNOWN IMAGE - Rejected")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    results = {}

    for category, config in CONFIGS.items():
        results[category] = prepare_category(category, config)

    print("\n" + "=" * 65)
    print("UNKNOWN IMAGE TESTING")
    print("=" * 65)

    print(
        "\nEnter image paths for each category."
        "\nYou can enter multiple paths separated by semicolons."
        "\nPress Enter to skip a category."
    )

    for category, data in results.items():

        print("\n", category)

        paths = input(
            "Enter image path(s): "
        ).strip()

        if not paths:
            continue

        for image_path in paths.split(";"):
            test_image(category, image_path.strip(), data)

    print("\nTesting completed.")
