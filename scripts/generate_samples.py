"""Create synthetic OK/NG panel images for the defect demo."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OK_DIR = ROOT / "data" / "samples" / "ok"
NG_DIR = ROOT / "data" / "samples" / "ng"


def make_ok(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = np.full((240, 320, 3), 210, dtype=np.uint8)
    noise = rng.integers(-8, 9, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (40, 40), (280, 200), (180, 180, 180), 2)
    cv2.imwrite(str(path), img)


def make_ng(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = np.full((240, 320, 3), 210, dtype=np.uint8)
    noise = rng.integers(-8, 9, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (40, 40), (280, 200), (180, 180, 180), 2)
    # Dark defect blobs
    for _ in range(int(rng.integers(1, 4))):
        cx = int(rng.integers(60, 260))
        cy = int(rng.integers(60, 180))
        axes = (int(rng.integers(8, 22)), int(rng.integers(6, 18)))
        cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, (40, 40, 40), -1)
    cv2.imwrite(str(path), img)


def main() -> None:
    OK_DIR.mkdir(parents=True, exist_ok=True)
    NG_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        make_ok(OK_DIR / f"ok_{i+1:02d}.png", seed=100 + i)
        make_ng(NG_DIR / f"ng_{i+1:02d}.png", seed=200 + i)
    print(f"Wrote samples to {OK_DIR} and {NG_DIR}")


if __name__ == "__main__":
    main()
