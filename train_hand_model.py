import csv
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import hand_features as hf

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "output" / "hand_landmark_dataset.csv"
MODEL_PATH = BASE_DIR / "output" / "hand_landmark_model.joblib"
FEATURE_COLUMNS = [
    f"landmark_{index}_{coordinate}"
    for index in range(21)
    for coordinate in ("x", "y", "z")
]
# Optional CSV columns. If you ever record them, the script uses them automatically:
#   group      -> person/session id; enables true group-aware cross-validation
#   handedness -> "Left"/"Right"; left hands are mirrored so everything looks right-handed
GROUP_COLUMN = "group"
HANDEDNESS_COLUMN = "handedness"

N_FOLDS = 5
RANDOM_STATE = 42
N_AUGMENT_COPIES = 4      # extra randomly-perturbed copies of each training row
MIRROR_AUGMENT = True    # True if you want the model to also handle the opposite hand
MAX_ROTATE_Z_DEG = 15.0
MAX_ROTATE_XY_DEG = 10.0
AXIS_SCALE_RANGE = 0.08
JITTER_STD = 0.01
SHUFFLE_WARNING_RATIO = 0.75


# --------------------------------------------------------------------------- data


def load_dataset():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Landmark dataset not found: {DATASET_PATH}")

    raw_rows: list[list[float]] = []
    labels: list[str] = []
    groups: list[str] = []
    handedness: list[str] = []

    with DATASET_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = [c for c in ["label", *FEATURE_COLUMNS] if c not in fieldnames]
        if missing:
            raise ValueError(
                f"CSV is missing required columns (first few): {missing[:5]}"
            )
        has_groups = GROUP_COLUMN in fieldnames
        has_hand = HANDEDNESS_COLUMN in fieldnames

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
            raw_rows.append(values)
            labels.append(label)
            if has_groups:
                groups.append(row[GROUP_COLUMN])
            if has_hand:
                handedness.append(row[HANDEDNESS_COLUMN].strip().lower())

    if not raw_rows:
        raise ValueError(f"No training rows found in {DATASET_PATH}.")

    points = hf.normalize_landmarks(np.asarray(raw_rows))
    if has_hand:
        left = np.array([h == "left" for h in handedness])
        points[left] = hf.mirror_landmarks(points[left])

    return (
        points,
        np.asarray(labels),
        np.asarray(groups) if has_groups else None,
        has_hand,
    )


def looks_shuffled(points: np.ndarray, labels: np.ndarray) -> float:
    """Ratio of (distance between consecutive same-label rows) to (distance between
    random same-label rows). Near 0 = rows are in capture order; near 1 = shuffled."""
    rng = np.random.default_rng(RANDOM_STATE)
    flat = points.reshape(len(points), -1)
    ratios = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 4:
            continue
        adjacent = np.linalg.norm(flat[idx[1:]] - flat[idx[:-1]], axis=1).mean()
        shuffled = rng.permutation(idx)
        random_pairs = np.linalg.norm(flat[idx] - flat[shuffled], axis=1).mean()
        if random_pairs > 0:
            ratios.append(adjacent / random_pairs)
    return float(np.mean(ratios)) if ratios else 1.0


def assign_folds(labels, groups, n_folds):
    """Return an int fold id per row.

    With a group column: StratifiedGroupKFold (whole people/sessions held out).
    Without one: per label, split rows into contiguous blocks in file order, so a
    validation fold is a stretch of time the model never saw. This only works if
    the CSV is still in capture order (checked in main).
    """
    folds = np.zeros(len(labels), dtype=int)
    if groups is not None:
        splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE
        )
        for fold, (_, test_idx) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups)
        ):
            folds[test_idx] = fold
        return folds

    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        folds[idx] = (np.arange(len(idx)) * n_folds) // len(idx)
    return folds


# ------------------------------------------------------------------- augmentation


def _rotation_matrices(rng, count):
    rz = np.deg2rad(rng.uniform(-MAX_ROTATE_Z_DEG, MAX_ROTATE_Z_DEG, count))
    rx = np.deg2rad(rng.uniform(-MAX_ROTATE_XY_DEG, MAX_ROTATE_XY_DEG, count))
    ry = np.deg2rad(rng.uniform(-MAX_ROTATE_XY_DEG, MAX_ROTATE_XY_DEG, count))
    zeros, ones = np.zeros(count), np.ones(count)
    cz, sz, cx, sx, cy, sy = np.cos(rz), np.sin(rz), np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry)
    mat_z = np.stack([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], axis=1).reshape(-1, 3, 3)
    mat_x = np.stack([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], axis=1).reshape(-1, 3, 3)
    mat_y = np.stack([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], axis=1).reshape(-1, 3, 3)
    return mat_z @ mat_y @ mat_x


def perturb(points, rng):
    """Small random rotation, per-axis stretch and jitter on normalized landmarks."""
    count = len(points)
    rotated = np.einsum("nij,nkj->nki", _rotation_matrices(rng, count), points)
    stretch = rng.uniform(1 - AXIS_SCALE_RANGE, 1 + AXIS_SCALE_RANGE, (count, 1, 3))
    noisy = rotated * stretch + rng.normal(0.0, JITTER_STD, rotated.shape)
    return hf.normalize_landmarks(noisy)


def build_training_matrix(points, labels, rng):
    parts, part_labels = [points], [labels]
    for _ in range(N_AUGMENT_COPIES):
        parts.append(perturb(points, rng))
        part_labels.append(labels)
    if MIRROR_AUGMENT:
        parts.append(hf.mirror_landmarks(points))
        part_labels.append(labels)
    stacked = np.concatenate(parts)
    features = hf.landmarks_to_features(stacked, normalized=True)
    return features, np.concatenate(part_labels)


# ------------------------------------------------------------------------ models


def candidate_models():
    return {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=1e-3,
                early_stopping=True,
                max_iter=300,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def cross_validate(
    name,
    factory,
    points,
    labels,
    folds,
    progress_callback: Callable[[str, bool], None],
):
    rng = np.random.default_rng(RANDOM_STATE)
    out_of_fold = np.empty(len(labels), dtype=labels.dtype)
    accuracies, macro_f1s = [], []
    fold_ids = np.unique(folds)
    for fold_number, fold in enumerate(fold_ids, start=1):
        progress_detail = f"{name} fold {fold_number}/{len(fold_ids)}"
        progress_callback(progress_detail, False)
        train, test = folds != fold, folds == fold
        x_train, y_train = build_training_matrix(points[train], labels[train], rng)
        x_test = hf.landmarks_to_features(points[test], normalized=True)
        model = factory()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        out_of_fold[test] = predictions
        accuracies.append(accuracy_score(labels[test], predictions))
        macro_f1s.append(f1_score(labels[test], predictions, average="macro", zero_division=0))
        progress_callback(progress_detail, True)
    print(
        f"{name:<24} accuracy {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}   "
        f"macro F1 {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}"
    )
    return float(np.mean(macro_f1s)), out_of_fold


# -------------------------------------------------------------------------- main


def train_model() -> None:
    points, labels, groups, has_handedness = load_dataset()
    class_counts = Counter(labels.tolist())
    smallest = min(class_counts.values())
    if smallest < 2:
        raise ValueError("Each label needs at least two rows.")
    n_folds = min(N_FOLDS, smallest)

    print(f"Rows: {len(labels)}   Labels: {len(class_counts)}   Folds: {n_folds}")
    if groups is not None:
        print(f"Using '{GROUP_COLUMN}' column: {len(np.unique(groups))} groups held out together.")
    else:
        ratio = looks_shuffled(points, labels)
        print(f"No '{GROUP_COLUMN}' column: using contiguous-block folds per label (file order).")
        if ratio > SHUFFLE_WARNING_RATIO:
            print(
                f"WARNING: consecutive rows look as different as random rows "
                f"(ratio {ratio:.2f}); the CSV may be shuffled, in which case block folds "
                "cannot stop near-duplicate leakage and scores will be optimistic. "
                "Record a person/session id per row for trustworthy numbers."
            )
        else:
            print(f"Rows look like capture order (ratio {ratio:.2f}); block folds are meaningful.")
    if not has_handedness:
        print(f"No '{HANDEDNESS_COLUMN}' column: assuming all samples are the same hand.")

    folds = assign_folds(labels, groups, n_folds)

    results = {}
    models = candidate_models()
    total_steps = len(models) * len(np.unique(folds)) + 1
    completed_steps = 0

    def update_progress(detail: str, completed: bool) -> None:
        nonlocal completed_steps
        if completed:
            completed_steps += 1
        progress = completed_steps / total_steps
        bar_width = 30
        filled_width = int(progress * bar_width)
        bar = "=" * filled_width + " " * (bar_width - filled_width)
        sys.stderr.write(
            f"\rTraining models: [{bar}] {progress:.0%} "
            f"({completed_steps}/{total_steps}) {detail:<32.32}"
        )
        sys.stderr.flush()

    update_progress("Starting cross-validation", False)
    try:
        for name, factory in models.items():
            score, out_of_fold = cross_validate(
                name, factory, points, labels, folds, update_progress
            )
            results[name] = (score, out_of_fold)

        sys.stderr.write("\n")
        sys.stderr.flush()

        best_name = max(results, key=lambda key: results[key][0])
        best_oof = results[best_name][1]
        print(f"\nBest by macro F1: {best_name}\n")
        print(
            classification_report(
                labels, best_oof, labels=sorted(class_counts), zero_division=0
            )
        )

        confusions = Counter(
            (true, pred) for true, pred in zip(labels, best_oof) if true != pred
        )
        if confusions:
            print("Most common mistakes (true -> predicted):")
            for (true, pred), count in confusions.most_common(10):
                print(f"  {true} -> {pred}: {count}")

        # Final model: fit on everything (with augmentation).
        rng = np.random.default_rng(RANDOM_STATE)
        x_all, y_all = build_training_matrix(points, labels, rng)
        final_model = models[best_name]()
        update_progress("Fitting final model", False)
        final_model.fit(x_all, y_all)
        update_progress("Final model complete", True)
    finally:
        sys.stderr.write("\n")
        sys.stderr.flush()

    joblib.dump(
        {
            "model": final_model,
            "model_name": best_name,
            "labels": sorted(class_counts),
            "input_columns": FEATURE_COLUMNS,
            "feature_version": hf.FEATURE_VERSION,
            "num_features": hf.NUM_FEATURES,
            "preprocessing": {
                "origin": "wrist (landmark 0)",
                "scale": "wrist to middle-finger MCP (landmark 9) distance",
                "left_hands_mirrored": has_handedness,
                "mirror_augment": MIRROR_AUGMENT,
            },
            "cv_macro_f1": {name: score for name, (score, _) in results.items()},
            "versions": {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "joblib": joblib.__version__,
            },
        },
        MODEL_PATH,
    )
    print(f"\nSaved {best_name} to {MODEL_PATH}")
    print(
        "At inference: features = hand_features.landmarks_to_features(landmarks63, "
        "mirror=<is_left_hand and left hands were mirrored in training>)"
    )


if __name__ == "__main__":
    train_model()