"""Landmark preprocessing shared by training AND inference.

Import `landmarks_to_features` in your inference code so the model sees
exactly the same features it was trained on.
"""
from __future__ import annotations

import numpy as np

FEATURE_VERSION = 1
NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_FINGER_MCP = 9
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),      # thumb
    (0, 5, 6, 7, 8),      # index
    (0, 9, 10, 11, 12),   # middle
    (0, 13, 14, 15, 16),  # ring
    (0, 17, 18, 19, 20),  # pinky
)
DISTANCE_POINTS = (0, 4, 8, 12, 16, 20)  # wrist + five fingertips
NUM_FEATURES = NUM_LANDMARKS * 3 + 15 + 15  # coords + distances + angles
_EPS = 1e-6


def to_points(landmarks) -> np.ndarray:
    """Accept (63,), (n, 63) or (n, 21, 3); return (n, 21, 3) float64."""
    array = np.asarray(landmarks, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    return array.reshape(-1, NUM_LANDMARKS, 3)


def normalize_landmarks(landmarks) -> np.ndarray:
    """Wrist becomes the origin; divide by wrist -> middle-finger-MCP distance.

    Removes position in the frame and hand size / camera distance. Rotation is
    deliberately kept, because orientation distinguishes some letters.
    """
    points = to_points(landmarks)
    points = points - points[:, WRIST : WRIST + 1, :]
    scale = np.linalg.norm(points[:, MIDDLE_FINGER_MCP, :], axis=1)
    return points / np.maximum(scale, _EPS)[:, None, None]


def mirror_landmarks(points: np.ndarray) -> np.ndarray:
    """Flip a (normalized, wrist-centered) hand left <-> right."""
    mirrored = points.copy()
    mirrored[..., 0] *= -1.0
    return mirrored


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    columns = []
    for i, a in enumerate(DISTANCE_POINTS):
        for b in DISTANCE_POINTS[i + 1 :]:
            columns.append(np.linalg.norm(points[:, a] - points[:, b], axis=1))
    return np.stack(columns, axis=1)


def _joint_angles(points: np.ndarray) -> np.ndarray:
    columns = []
    for chain in FINGER_CHAINS:
        for i in range(1, len(chain) - 1):
            u = points[:, chain[i - 1]] - points[:, chain[i]]
            v = points[:, chain[i + 1]] - points[:, chain[i]]
            denom = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
            cosine = np.sum(u * v, axis=1) / np.maximum(denom, _EPS)
            columns.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.stack(columns, axis=1)


def landmarks_to_features(
    landmarks, *, normalized: bool = False, mirror: bool = False
) -> np.ndarray:
    """Raw landmarks -> (n, NUM_FEATURES) feature matrix.

    normalized: set True if `landmarks` already went through normalize_landmarks.
    mirror: set True to flip the hand (use for left hands if you train on right).
    """
    points = to_points(landmarks)
    if not normalized:
        points = normalize_landmarks(points)
    if mirror:
        points = mirror_landmarks(points)
    return np.concatenate(
        [
            points.reshape(len(points), -1),
            _pairwise_distances(points),
            _joint_angles(points),
        ],
        axis=1,
    )