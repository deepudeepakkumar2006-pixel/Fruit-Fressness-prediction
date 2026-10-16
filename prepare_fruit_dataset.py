import os
import shutil
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

OLD_BANANA_DATASET = r"D:\pbl food freshness detection dataset"
NEW_DATASET_OUTPUT = r"D:\pbl food freshness multi dataset"

# ============================================================
# 1. CREATE FINAL DATASET FOLDERS
# ============================================================

splits = ["train", "valid", "test"]

banana_classes = [
    "unripe",
    "ripe",
    "overripe",
    "rotten"
]

other_fruits = [
    "apple",
    "grape",
    "guava",
    "jujube",
    "orange",
    "pomegranate",
    "strawberry"
]

freshness_classes = ["fresh", "rotten"]

print("=" * 60)
print("FOOD FRESHNESS MULTI-FRUIT DATASET PREPARATION")
print("=" * 60)

# Create folders
for split in splits:
    split_path = os.path.join(NEW_DATASET_OUTPUT, split)
    os.makedirs(split_path, exist_ok=True)

# ============================================================
# 2. COPY EXISTING BANANA DATASET
# ============================================================

print("\n[1/3] Copying existing banana dataset...")

for split in splits:

    source_split = os.path.join(OLD_BANANA_DATASET, split)
    destination_split = os.path.join(NEW_DATASET_OUTPUT, split)

    if not os.path.exists(source_split):
        print(f"WARNING: {source_split} not found!")
        continue

    for class_name in banana_classes:

        source_class = os.path.join(source_split, class_name)
        destination_class = os.path.join(destination_split, class_name)

        os.makedirs(destination_class, exist_ok=True)

        if not os.path.exists(source_class):
            print(f"WARNING: {source_class} not found!")
            continue

        files = os.listdir(source_class)

        for filename in files:

            source_file = os.path.join(source_class, filename)
            destination_file = os.path.join(destination_class, filename)

            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)

        print(
            f"Copied banana/{class_name} "
            f"from {split}: {len(files)} images"
        )

# ============================================================
# 3. LOAD NEW MULTI-FRUIT DATASET
# ============================================================

print("\n[2/3] Loading new fruit dataset...")

dataset = load_dataset(
    "Project-AgML/fresh_rotten_fruit_classification",
    "raw"
)

train_data = dataset["train"]

print(f"Total new dataset images: {len(train_data)}")

# ============================================================
# FIND LABEL NAMES
# ============================================================

label_feature = train_data.features["label"]

if hasattr(label_feature, "names"):
    label_names = label_feature.names
else:
    label_names = ["fresh", "rotten"]

print("Label names:", label_names)

# ============================================================
# FILTER OUT BANANAS
# ============================================================

print("\nRemoving banana images from the new dataset...")

filtered_indices = []

for i in range(len(train_data)):

    fruit = str(train_data[i]["fruit_type"]).lower()

    if fruit != "banana":
        filtered_indices.append(i)

print(f"Non-banana images: {len(filtered_indices)}")

# ============================================================
# CREATE CLASS INFORMATION
# ============================================================

records = []

for index in filtered_indices:

    item = train_data[index]

    fruit = str(item["fruit_type"]).lower()

    label_value = item["label"]

    # Convert numerical label to text
    if isinstance(label_value, int):
        freshness = label_names[label_value].lower()
    else:
        freshness = str(label_value).lower()

    class_name = f"{fruit}_{freshness}"

    records.append({
        "index": index,
        "class_name": class_name
    })

# ============================================================
# CHECK CLASSES
# ============================================================

print("\nNew classes found:")

class_counts = {}

for record in records:

    class_name = record["class_name"]

    class_counts[class_name] = class_counts.get(class_name, 0) + 1

for class_name in sorted(class_counts):
    print(f"{class_name}: {class_counts[class_name]} images")

# ============================================================
# 4. SPLIT NEW DATASET
# ============================================================

print("\n[3/3] Splitting new fruit images...")

indices = [record["index"] for record in records]
classes = [record["class_name"] for record in records]

# 70% training, 15% validation, 15% testing

train_indices, temp_indices, train_classes, temp_classes = train_test_split(
    indices,
    classes,
    test_size=0.30,
    random_state=42,
    stratify=classes
)

valid_indices, test_indices, valid_classes, test_classes = train_test_split(
    temp_indices,
    temp_classes,
    test_size=0.50,
    random_state=42,
    stratify=temp_classes
)

print(f"New training images: {len(train_indices)}")
print(f"New validation images: {len(valid_indices)}")
print(f"New testing images: {len(test_indices)}")

# ============================================================
# FUNCTION TO SAVE IMAGES
# ============================================================

def save_images(image_indices, split_name):

    print(f"\nSaving {split_name} images...")

    for count, index in enumerate(image_indices, start=1):

        item = train_data[index]

        fruit = str(item["fruit_type"]).lower()

        label_value = item["label"]

        if isinstance(label_value, int):
            freshness = label_names[label_value].lower()
        else:
            freshness = str(label_value).lower()

        class_name = f"{fruit}_{freshness}"

        class_folder = os.path.join(
            NEW_DATASET_OUTPUT,
            split_name,
            class_name
        )

        os.makedirs(class_folder, exist_ok=True)

        image = item["image"]

        filename = f"{fruit}_{freshness}_{index:05d}.jpg"

        output_file = os.path.join(
            class_folder,
            filename
        )

        image.convert("RGB").save(
            output_file,
            "JPEG",
            quality=90
        )

        if count % 100 == 0:
            print(f"Saved {count} / {len(image_indices)}")

# ============================================================
# SAVE NEW FRUIT IMAGES
# ============================================================

save_images(train_indices, "train")
save_images(valid_indices, "valid")
save_images(test_indices, "test")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETED!")
print("=" * 60)

print("\nFinal dataset location:")
print(NEW_DATASET_OUTPUT)

print("\nExpected classes:")

final_classes = banana_classes.copy()

for fruit in other_fruits:
    final_classes.append(f"{fruit}_fresh")
    final_classes.append(f"{fruit}_rotten")

for number, class_name in enumerate(final_classes, start=1):
    print(f"{number}. {class_name}")

print(f"\nTOTAL CLASSES: {len(final_classes)}")

print("\nYour original banana dataset was NOT modified.")
print("The combined dataset was created separately.")
print("\n" + "=" * 60)
print("CHECKING FINAL DATASET")
print("=" * 60)

for split in ["train", "valid", "test"]:

    print(f"\n{split.upper()}:")

    split_path = os.path.join(NEW_DATASET_OUTPUT, split)

    total = 0

    for class_name in sorted(os.listdir(split_path)):

        class_path = os.path.join(split_path, class_name)

        if os.path.isdir(class_path):

            image_count = len([
                file
                for file in os.listdir(class_path)
                if file.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            print(f"{class_name}: {image_count}")

            total += image_count

    print(f"TOTAL {split.upper()} IMAGES: {total}")