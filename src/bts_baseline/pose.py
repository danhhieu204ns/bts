from __future__ import annotations

import math

from .colmap_io import camera_center, qvec_normalize


Vec3 = tuple[float, float, float]
QVec = tuple[float, float, float, float]


def euclidean(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def qvec_angle(a: QVec, b: QVec) -> float:
    qa = qvec_normalize(a)
    qb = qvec_normalize(b)
    dot = abs(sum(x * y for x, y in zip(qa, qb)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def pose_distance(
    train_qvec: QVec,
    train_tvec: Vec3,
    test_qvec: QVec,
    test_tvec: Vec3,
    orientation_weight: float,
) -> tuple[float, float, float]:
    train_center = camera_center(train_qvec, train_tvec)
    test_center = camera_center(test_qvec, test_tvec)
    center_distance = euclidean(train_center, test_center)
    angle = qvec_angle(train_qvec, test_qvec)
    score = center_distance + orientation_weight * angle
    return score, center_distance, angle
