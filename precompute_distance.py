import dither
import colour.difference
import numpy as np

COLOURS = 256


def rgb_to_lab(rgb: np.ndarray):
    srgb = np.clip(
        dither.linear_to_srgb_array(rgb.astype(np.float32) / 255), 0.0,
        1.0)
    xyz = colour.sRGB_to_XYZ(srgb)
    return colour.XYZ_to_Lab(xyz)


def nearest_colours():
    diffs = np.empty((COLOURS ** 3, 16), dtype=np.float32)
    all_rgb = np.array(tuple(np.ndindex(COLOURS, COLOURS, COLOURS)),
                       dtype=np.uint8)
    all_lab = rgb_to_lab(all_rgb)

    for i, palette_rgb in dither.RGB.items():
        print(i)
        palette_lab = rgb_to_lab(palette_rgb)
        diffs[:, i] = colour.difference.delta_E_CIE2000(all_lab, palette_lab)

    norm = np.max(diffs)
    return (diffs / norm * 255).astype(np.uint8)


n = nearest_colours()
out = np.memmap(filename="distances.npy", mode="w+", dtype=np.uint8,
                shape=n.shape)
out[:] = n[:]
