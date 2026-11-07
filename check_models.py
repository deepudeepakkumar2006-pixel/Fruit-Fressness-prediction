import os
from tensorflow.keras.models import load_model

models = [
    "models/food_freshness_18class_model.keras",
    "models/vegetable_freshness_model.keras"
]

datasets = [
    r"D:\pbl food freshness multi dataset",
    r"D:\pbl food freshness project\vegetable_split_dataset"
]

for path in models:
    print("\n" + "=" * 50)
    print("MODEL:", path)

    if not os.path.exists(path):
        print("MODEL NOT FOUND!")
        continue

    model = load_model(path)

    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    print("Global Average Pooling layers:")
    for layer in model.layers:
        if "GlobalAveragePooling2D" in layer.__class__.__name__:
            print("-", layer.name)

    print("Last 6 layers:")
    for layer in model.layers[-6:]:
        print(layer.name, "-", layer.__class__.__name__)

print("\n" + "=" * 50)
print("DATASET PATH CHECK")

for path in datasets:
    print(path)
    print("EXISTS" if os.path.isdir(path) else "NOT FOUND")