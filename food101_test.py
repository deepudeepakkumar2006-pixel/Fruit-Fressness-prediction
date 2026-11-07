
import os
from transformers import pipeline

MODEL_NAME = "nateraw/food"

print("Loading Food-101 pretrained model...")
print("The first download is approximately 344 MB.")

classifier = pipeline(
    "image-classification",
    model=MODEL_NAME,
    device=-1
)

print("\nFood-101 model loaded successfully!")


def test_image(path, category):

    if not os.path.isfile(path):
        print("\nImage not found:", path)
        return

    print("\n" + "=" * 55)
    print("IMAGE:", os.path.basename(path))
    print("SELECTED CATEGORY:", category)

    predictions = classifier(path, top_k=5)

    print("\nFOOD-101 PREDICTIONS:")

    for i, result in enumerate(predictions, 1):
        print(
            f"{i}. {result['label']} - "
            f"{result['score'] * 100:.2f}%"
        )


if __name__ == "__main__":

    for category in ["FRUITS", "VEGETABLES"]:

        print("\n" + "#" * 55)
        print(category)

        paths = input(
            "Enter image paths separated by semicolons: "
        ).strip()

        if paths:
            for path in paths.split(";"):
                test_image(path.strip(), category)

    print("\nFood-101 testing completed.")
