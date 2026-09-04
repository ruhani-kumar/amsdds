import numpy as np


def shades_of_gray(arr: np.ndarray, power: int = 6) -> np.ndarray:
    """Normalise illumination. Input/output are uint8 HxWx3 RGB arrays."""
    a = arr.astype(np.float32)
    vec = np.power(np.mean(np.power(a, power), axis=(0, 1)), 1.0 / power)
    vec = vec / (np.sqrt(np.sum(vec ** 2)) + 1e-8)
    return np.clip(a / (vec * np.sqrt(3) + 1e-8), 0, 255).astype(np.uint8)
