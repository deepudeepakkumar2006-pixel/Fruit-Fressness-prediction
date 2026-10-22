
import random
import shutil
from pathlib import Path

# Original dataset containing the 10 class folders
SOURCE_DIR = Path(r"D:\pbl food freshness project\vegetable_dataset")

# New folder for training, validation, and testing images
OUTPUT_DIR = Path(r"D:\pbl food freshness project\vegetable_split_dataset")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42

def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source folder not found:\n{SOURCE_DIR}")
        return

    if OUTPUT_DIR.exists():
        print(f"ERROR: Output folder already exists:\n{OUTPUT_DIR}")
        print("Rename or remove the output folder before running again.")
        print("Your original dataset has not been changed.")
        return

    class_folders = sorted(
        folder for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    )

    if len(class_folders) != 10:
        print(f"ERROR: Expected 10 class folders, found {len(class_folders)}.")
        print("Check the vegetable_dataset folder before continuing.")
        return

    random.seed(RANDOM_SEED)
    total_images = 0

    for class_folder in class_folders:
        images = sorted(
            file for file in class_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) < 3:
            print(f"ERROR: Too few images in {class_folder.name}")
            print("No dataset was created. Check this class folder.")
            return

    # Create output only after checking all 10 classes.
    for split in ("train", "valid", "test"):
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    print("\nCreating vegetable dataset split...\n")

    for class_folder in class_folders:
        images = sorted(
            file for file in class_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        random.shuffle(images)
        count = len(images)

        train_count = int(count * TRAIN_RATIO)
        valid_count = int(count * VALID_RATIO)

        train_images = images[:train_count]
        valid_images = images[train_count:train_count + valid_count]
        test_images = images[train_count + valid_count:]

        split_images = {
            "train": train_images,
            "valid": valid_images,
            "test": test_images,
        }

        for split, files in split_images.items():
            destination = OUTPUT_DIR / split / class_folder.name
            destination.mkdir(parents=True, exist_ok=True)

            for image_file in files:
                shutil.copy2(image_file, destination / image_file.name)

        total_images += count

        print(
            f"{class_folder.name}: "
            f"train={len(train_images)}, "
            f"valid={len(valid_images)}, "
            f"test={len(test_images)}"
        )

    print("\nDataset split completed successfully!")
    print(f"Total images copied: {total_images}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Original dataset remains unchanged.")

if __name__ == "__main__":
    main()
