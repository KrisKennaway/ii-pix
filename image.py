import numpy as np
from PIL import Image


def srgb_to_linear_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** gamma)


def linear_to_srgb_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1.0 / gamma) -
                    0.055)


def srgb_to_linear(im: np.ndarray) -> np.ndarray:
    rgb_linear = srgb_to_linear_array(im / 255.0, gamma=2.4)
    return (np.clip(rgb_linear, 0.0, 1.0) * 255).astype(np.float32)


def linear_to_srgb(im: np.ndarray) -> np.ndarray:
    srgb = linear_to_srgb_array(im / 255.0, gamma=2.4)
    return (np.clip(srgb, 0.0, 1.0) * 255).astype(np.float32)


def open(x_res:int, y_res:int, filename: str) -> np.ndarray:
    im = Image.open(filename)
    # TODO: convert to sRGB colour profile explicitly, in case it has some other
    #  profile already.
    if im.mode != "RGB":
        im = im.convert("RGB")

    # Convert to linear RGB before rescaling so that colour interpolation is
    # in linear space
    linear = srgb_to_linear(np.asarray(im)).astype(np.uint8)
    rescaled = Image.fromarray(linear).resize((x_res, y_res), Image.LANCZOS)
    # TODO: better performance with malloc'ed array?
    return np.array(rescaled).astype(np.float32)

