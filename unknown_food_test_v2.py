
import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import (
    image_dataset_from_directory,
    load_img,
    img_to_array
)

BASE_DIR = r"D:\pbl food freshness project"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
REFERENCE_LIMIT = 100

CONFIGS = {
    "FRUITS": {
        "model": os.path.join(
            BASE_DIR, "models", "food_freshness_18class_model.keras"
        ),
        "dataset": r"D:\pbl food freshness multi dataset"
    },
    "VEGETABLES": {
        "model": os.path.join(
            BASE_DIR, "models", "vegetable_freshness_model.keras"
        ),
        "dataset": os.path.join(BASE_DIR, "vegetable_split_dataset")
    }
}


def normalize(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-10)


def prepare_model(category, config):

    print("\n" + "=" * 65)
    print("LOADING:", category)
    print("=" * 65)

    model = load_model(config["model"])

    feature_model = Model(
        inputs=model.input,
        outputs=model.get_layer(
            "global_average_pooling2d"
        ).output
    )

    train_ds = image_dataset_from_directory(
        os.path.join(config["dataset"], "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    valid_ds = image_dataset_from_directory(
        os.path.join(config["dataset"], "valid"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names

    if valid_ds.class_names != class_names:
        raise ValueError("Training and validation classes do not match.")

    if len(class_names) != model.output_shape[-1]:
        raise ValueError("Model output and dataset class counts differ.")

    print("Classes:", class_names)

    # Extract training features and build class references.
    print("\nExtracting training features...")

    train_features = []
    train_labels = []

    for images, labels in train_ds:
        features = feature_model.predict(images, verbose=0)
        train_features.append(features)
        train_labels.append(labels.numpy())

    train_features = normalize(np.concatenate(train_features))
    train_labels = np.concatenate(train_labels)

    centroids = []

    for index in range(len(class_names)):
        class_features = train_features[
            train_labels == index
        ][:REFERENCE_LIMIT]

        centroid = np.mean(class_features, axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-10)
        centroids.append(centroid)

    centroids = np.asarray(centroids)

    # Evaluate classifier and feature agreement on validation images.
    print("\nChecking validation images...")

    actual_labels = []
    classifier_predictions = []
    feature_predictions = []
    correct_class_similarities = []

    for images, labels in valid_ds:

        probabilities = model.predict(images, verbose=0)
        features = feature_model.predict(images, verbose=0)
        features = normalize(features)

        similarities = features @ centroids.T

        actual = labels.numpy()
        classifier_pred = np.argmax(probabilities, axis=1)
        feature_pred = np.argmax(similarities, axis=1)

        correct_similarity = similarities[
            np.arange(len(actual)), actual
        ]

        actual_labels.extend(actual)
        classifier_predictions.extend(classifier_pred)
        feature_predictions.extend(feature_pred)
        correct_class_similarities.extend(correct_similarity)

    actual_labels = np.asarray(actual_labels)
    classifier_predictions = np.asarray(classifier_predictions)
    feature_predictions = np.asarray(feature_predictions)
    correct_class_similarities = np.asarray(
        correct_class_similarities
    )

    classifier_accuracy = np.mean(
        classifier_predictions == actual_labels
    ) * 100

    feature_accuracy = np.mean(
        feature_predictions == actual_labels
    ) * 100

    agreement = np.mean(
        classifier_predictions == feature_predictions
    ) * 100

    print("\nVALIDATION DIAGNOSTICS")
    print("Validation images:", len(actual_labels))
    print("Classifier top-1 accuracy:", round(classifier_accuracy, 2), "%")
    print("Feature nearest-class accuracy:", round(feature_accuracy, 2), "%")
    print("Classifier/feature agreement:", round(agreement, 2), "%")

    print("\nPer-class similarity ranges:")

    for index, name in enumerate(class_names):
        values = correct_class_similarities[
            actual_labels == index
        ]

        if len(values):
            print(
                name,
                "| 5th percentile:",
                round(float(np.percentile(values, 5)), 4),
                "| median:",
                round(float(np.median(values)), 4)
            )

    return {
        "model": model,
        "feature_model": feature_model,
        "centroids": centroids,
        "class_names": class_names
    }


def inspect_image(category, image_path, data):

    if not os.path.isfile(image_path):
        print("\nIMAGE NOT FOUND:", image_path)
        return

    image = load_img(image_path, target_size=IMG_SIZE)
    array = img_to_array(image)
    array = np.expand_dims(array, axis=0)

    probabilities = data["model"].predict(array, verbose=0)[0]

    features = data["feature_model"].predict(array, verbose=0)
    features = normalize(features)

    similarities = (features @ data["centroids"].T)[0]

    classifier_index = int(np.argmax(probabilities))
    feature_index = int(np.argmax(similarities))

    top_indices = np.argsort(probabilities)[::-1][:3]

    print("\n" + "-" * 60)
    print("IMAGE:", os.path.basename(image_path))
    print("CATEGORY:", category)

    print("\nClassifier top predictions:")

    for index in top_indices:
        print(
            data["class_names"][index],
            "| Confidence:",
            round(float(probabilities[index] * 100), 2), "%"
        )

    print("\nFeature nearest class:",
          data["class_names"][feature_index])

    print("Feature similarity:",
          round(float(similarities[feature_index]), 4))

    print("Classifier predicted class:",
          data["class_names"][classifier_index])

    print(
        "Classifier and feature agreement:",
        "YES" if classifier_index == feature_index else "NO"
    )


if __name__ == "__main__":

    results = {}

    for category, config in CONFIGS.items():
        results[category] = prepare_model(category, config)

    print("\n" + "=" * 65)
    print("DIAGNOSTIC IMAGE TESTING")
    print("=" * 65)

    for category, data in results.items():

        print("\n", category)

        paths = input(
            "Enter image paths separated by semicolons (or press Enter): "
        ).strip()

        if paths:
            for path in paths.split(";"):
                inspect_image(category, path.strip(), data)

    print("\nDiagnostic testing completed.")
