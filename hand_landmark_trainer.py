import csv
import math
from collections import Counter
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "hand_landmark_dataset.csv"
MODEL_PATH = BASE_DIR / "hand_landmark_model.joblib"
FEATURE_COLUMNS = [
    f"landmark_{index}_{coordinate}"
    for index in range(21)
    for coordinate in ("x", "y", "z")
]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset() -> tuple[list[list[float]], list[str]]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Landmark dataset not found: {DATASET_PATH}")

    features: list[list[float]] = []
    labels: list[str] = []
    expected_columns = ["label", *FEATURE_COLUMNS]

    with DATASET_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"Unexpected CSV columns in {DATASET_PATH}; "
                "expected label and 63 ordered landmark coordinates."
            )

        for line_number, row in enumerate(reader, start=2):
            label = row["label"]
            if not label:
                raise ValueError(f"Missing label on CSV line {line_number}.")
            try:
                values = [float(row[column]) for column in FEATURE_COLUMNS]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid landmark value on CSV line {line_number}."
                ) from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite landmark value on CSV line {line_number}.")
            features.append(values)
            labels.append(label)

    if not features:
        raise ValueError(f"No training rows found in {DATASET_PATH}.")
    return features, labels


def train_model() -> None:
    features, labels = load_dataset()
    class_counts = Counter(labels)
    if min(class_counts.values()) < 2:
        raise ValueError(
            "Each label needs at least two rows for a stratified train/validation split."
        )

    x_train, x_validation, y_train, y_validation = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)

    print(f"Training rows: {len(x_train)}")
    print(f"Validation rows: {len(x_validation)}")
    print(f"Labels: {len(class_counts)}")
    print(f"Validation accuracy: {accuracy_score(y_validation, predictions):.4f}")
    print(f"Validation macro F1: {f1_score(y_validation, predictions, average='macro'):.4f}")
    print(
        classification_report(
            y_validation,
            predictions,
            labels=sorted(class_counts),
            zero_division=0,
        )
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "labels": sorted(class_counts),
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()