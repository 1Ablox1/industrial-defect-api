"""OpenCV pipeline: detect dark defect blobs on a light panel."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    area: float


@dataclass
class InspectResult:
    result: str  # OK | NG
    score: float
    boxes: list[dict]


def inspect_image(image_bgr: np.ndarray, min_area: int = 80) -> InspectResult:
    """Score defects via adaptive threshold + contour area.

    Synthetic OK panels are mostly uniform; NG panels contain dark blobs.
    score is in [0, 1] where higher means more defect-like.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Invert so dark defects become bright foreground
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    total_area = float(gray.shape[0] * gray.shape[1])
    defect_area = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append(Box(x=x, y=y, w=w, h=h, area=area))
        defect_area += area

    ratio = defect_area / total_area if total_area else 0.0
    # Map ratio to a 0–1 score; small blobs still flag NG
    score = float(min(1.0, ratio * 25.0 + (0.35 if boxes else 0.0)))
    label = "NG" if boxes else "OK"
    if not boxes:
        score = float(min(0.2, ratio * 10.0))

    return InspectResult(
        result=label,
        score=round(score, 4),
        boxes=[
            {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "area": round(b.area, 1)}
            for b in boxes
        ],
    )


def inspect_bytes(data: bytes) -> InspectResult:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return inspect_image(image)
