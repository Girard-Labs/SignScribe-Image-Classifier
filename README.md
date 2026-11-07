# SignScribe Image Classifier

This project is a hand-sign image classifier for ASL alphabet recognition. It uses MediaPipe hand landmarks, geometric feature extraction, and a machine learning model to predict letters from images of a hand.

## The Dataset
The original dataset can be downloaded from Kaggle: [asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). It consists of 50 x 50 px color images in which gestures of the A-Z symbols are displayed with hands on different background and lighting conditions.

## How it works

The pipeline is:

1. Read training images from the `training-data` directory.
2. Detect a hand with MediaPipe HandLandmarker.
3. Extract 21 hand landmarks, each with x/y/z coordinates.
4. Normalize those landmarks so the model is less sensitive to hand position, size, and distance from the camera.
5. Convert the normalized landmarks into engineered features such as:
   - flattened landmark coordinates
   - pairwise fingertip distances
   - joint-angle descriptors
6. Train a classifier to predict the label for each sign.
7. Save the trained model as a `.joblib` artifact for reuse.

## Project structure

- `hand_landmark_compiler.py`  
  Builds the dataset by processing every image in the training directories, showing a progress bar, and writing rows to `hand_landmark_dataset.csv`.

- `hand_features.py`  
  Defines the landmark normalization and feature extraction logic. This is the shared preprocessing used to keep training and inference consistent.

- `train_hand_model.py`  
  Loads the CSV dataset, performs cross-validation, augments data, trains candidate models, and saves the final model.

- `hand_landmark_dataset.csv`  
  Generated dataset with label plus 63 landmark coordinates for each training sample.

- `hand_landmark_model.joblib`  
  Trained model artifact.

- `hand_landmarker.task`  
  MediaPipe hand landmark model used by `hand_landmark_compiler.py`.

- `training-data/`  
  ASL alphabet image dataset, grouped by letter directories such as `A`, `B`, `C`, etc.

## Data flow

The actual flow looks like this:

- image -> OpenCV read
- image -> MediaPipe hand detection
- detected hand -> 21 landmark points
- landmark points -> normalized + feature-engineered vector
- feature vector -> classifier -> predicted letter

## Why normalization matters

The landmark coordinates are normalized relative to the wrist and a scale tied to the hand size. This helps the model focus on the shape and orientation of the hand rather than the camera framing or hand size.

## Training notes

The project includes robust training practices such as:

- group-aware and contiguous-block validation strategies
- augmentation with small rotations, scaling, and jitter
- left/right hand mirroring for consistency
- cross-validation with macro F1 scoring

These steps improve generalization beyond the raw training images.

## Important caveat

This repository is primarily a model-training pipeline, not a finished real-time app. It does not include a webcam-based UI or complete inference service out of the box. The main purpose is to prepare a dataset, train a classifier, and save a reusable model.

## Typical workflow

1. Ensure the training data is present in `training-data/`.
2. Run the dataset compiler:
   ```bash
   python hand_landmark_compiler.py
   ```
3. Train the model:
   ```bash
   python train_hand_model.py
   ```
4. Use the saved model artifact for downstream inference.

## Requirements

This project depends on:

- Python
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- joblib

## Summary

SignScribe Image Classifier combines MediaPipe hand landmark tracking with carefully engineered hand-shape features and machine learning to recognize ASL alphabet signs from static images.

## About The Model

The model is a sign-language alphabet classifier (A–Z plus `del`, `nothing`, `space`, which matches the ASL alphabet image datasets). The model was trained on 16 model/fold combinations, and the output shows a cross-validated comparison followed by a detailed report for the winner.

**The model comparison**

Each line reports the mean ± standard deviation across 5 folds:

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random forest | 0.9426 ± 0.0318 | 0.9307 ± 0.0295 |
| Hist. gradient boosting | 0.9264 ± 0.0382 | 0.9038 ± 0.0377 |
| MLP | 0.9443 ± 0.0316 | 0.9328 ± 0.0326 |

The MLP wins on macro F1, but only by about 0.002 over random forest. The standard deviations (~0.03) are roughly 15x larger than that gap, so the two are statistically indistinguishable. Gradient boosting is clearly a step behind.

**The classification report**

This is the MLP's out-of-fold performance across 71,473 samples. Overall it's 94% accurate, and most letters sit at 0.93-0.99 F1. Some details:

- **Best classes:** K, F, L (0.98-0.99). Their hand shapes are distinctive.
- **Weak spots:** M and N (~0.82 F1) are the clear problem, and `nothing` is at 0.65.
- **`nothing`** has only 87 samples versus ~2,500 for everything else, so its low precision (0.58) and recall (0.75) come from tiny support. Its score is noisy and barely moves the overall numbers, but macro F1 weights it equally with every other class, which drags that figure down. Accuracy (94%) and weighted avg are the more representative numbers here.
- **Macro vs. weighted avg:** both are ~0.93-0.94, which tells you class imbalance isn't hurting you much outside `nothing`.

**The confusion list**

This explains the weak classes:

- **M ↔ N (215 and 189):** the biggest error source by far. In ASL, M and N are nearly identical fists with the thumb tucked under three vs. two fingers, so this is a genuinely hard visual distinction, not just a model flaw.
- **R → U (180):** R is crossed fingers, U is two fingers together. Very similar shapes. It's one-directional (R is often called U, but not the reverse), which is why R's recall is just 0.90 while its precision is 0.96.
- **Q → P (75), I ↔ J (50/49), Z → X, D → O, T → A:** all plausible look-alike pairs. I/J differ mainly by motion in real ASL (J is a moving sign), which a still image can't capture.
- **N → del (53):** this one is odd, since del isn't a lookalike letter. It may be a labeling or data-capture issue worth inspecting.

**What I'd try next**

1. Look at example images (or landmarks) of M/N and R/U misclassifications to see if the features you're using can separate them. If you use raw pixels or hand landmarks, adding finger-specific geometry, or higher resolution crops around the thumb, may help.
2. Check for group leakage in your CV split.
3. For `nothing`, either collect more samples or exclude it from macro metrics when comparing models.
4. Since the top models are tied within noise, consider choosing on speed, simplicity, or calibration rather than the 0.002 F1 edge.