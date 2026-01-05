from __future__ import annotations

import io

import numpy as np
from PIL import Image


def load_image_bytes(data: bytes) -> np.ndarray:
    """
    Loads bytes into an OpenCV-style BGR uint8 image.
    We use PIL for robust decoding of varied screenshot formats.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    # RGB -> BGR for OpenCV conventions
    return rgb[:, :, ::-1].copy()

