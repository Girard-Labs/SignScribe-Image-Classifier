import csv
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_DIR = (
    BASE_DIR / "training-data" / "asl_alphabet_train" / "asl_alphabet_train"
)
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
OUTPUT_PATH = BASE_DIR / "hand_landmark_dataset.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
LANDMARK_COLUMNS = [
    f"landmark_{index}_{coordinate}"
    for index in range(21)
    for coordinate in ("x", "y", "z")
]


def compile_dataset() -> None:
    if not TRAINING_DATA_DIR.is_dir():
        raise FileNotFoundError(f"Training data directory not found: {TRAINING_DATA_DIR}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Hand landmark model not found: {MODEL_PATH}")

    class_directories = sorted(
        path for path in TRAINING_DATA_DIR.iterdir() if path.is_dir()
    )
    if not class_directories:
        raise FileNotFoundError(f"No class directories found in {TRAINING_DATA_DIR}")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    images_processed = 0
    rows_written = 0
    unreadable_images = 0
    images_without_hands = 0

    try:
        with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["label", *LANDMARK_COLUMNS])

            for class_directory in class_directories:
                image_paths = sorted(
                    path
                    for path in class_directory.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
                for image_path in image_paths:
                    images_processed += 1
                    image = cv2.imread(str(image_path))
                    if image is None:
                        unreadable_images += 1
                        continue

                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = detector.detect(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                    )
                    if not result.hand_landmarks:
                        images_without_hands += 1
                        continue

                    landmarks = result.hand_landmarks[0]
                    coordinates = [
                        value
                        for landmark in landmarks
                        for value in (landmark.x, landmark.y, landmark.z)
                    ]
                    writer.writerow([class_directory.name, *coordinates])
                    rows_written += 1
    finally:
        detector.close()

    print(
        f"Processed {images_processed} images; wrote {rows_written} rows to "
        f"{OUTPUT_PATH}. Skipped {images_without_hands} images with no detected "
        f"hand and {unreadable_images} unreadable images."
    )


if __name__ == "__main__":
    compile_dataset()
