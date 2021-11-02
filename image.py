"""Image transformation functions."""

import numpy as np
from PIL import Image


def srgb_to_linear_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** gamma)


def linear_to_srgb_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1.0 / gamma) -
                    0.055)


def srgb_to_linear(im: np.ndarray, gamma=2.4) -> np.ndarray:
    rgb_linear = srgb_to_linear_array(im / 255.0, gamma=gamma)
    return (np.clip(rgb_linear, 0.0, 1.0) * 255).astype(np.float32)


def linear_to_srgb(im: np.ndarray, gamma=2.4) -> np.ndarray:
    srgb = linear_to_srgb_array(im / 255.0, gamma=gamma)
    return (np.clip(srgb, 0.0, 1.0) * 255).astype(np.float32)


def open(filename: str) -> np.ndarray:
    im = Image.open(filename)
    # TODO: convert to sRGB colour profile explicitly, in case it has some other
    #  profile already.
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def resize(
        image: Image, x_res, y_res, gamma: float = 2.4,
        srgb_output: bool = False) -> Image:
    # Convert to linear RGB before rescaling so that colour interpolation is
    # in linear space
    linear = srgb_to_linear(np.asarray(image), gamma=gamma).astype(np.uint8)
    res = Image.fromarray(linear).resize((x_res, y_res), Image.LANCZOS)
    if srgb_output:
        return Image.fromarray(
            linear_to_srgb(np.array(res, dtype=np.float32), gamma=gamma).astype(
                np.uint8))
    else:
        return res
