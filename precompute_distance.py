import bz2
import pickle

import dither
import colour.difference
import numpy as np

COLOURS = 256


def nearest_colours():
    diffs = np.empty((COLOURS ** 3, 16), dtype=np.float32)

    all_rgb = np.array(tuple(np.ndindex(COLOURS, COLOURS, COLOURS)),
                       dtype=np.uint8)
    all_srgb = dither.linear_to_srgb_array(all_rgb / 255)
    all_xyz = colour.sRGB_to_XYZ(all_srgb)
    all_lab = colour.XYZ_to_Lab(all_xyz)

    for i, p in dither.LAB.items():
        print(i)
        diffs[:, i] = colour.difference.delta_E_CIE2000(all_lab, p)

    return diffs


n = nearest_colours()
with bz2.open("nearest.pickle.bz2", "wb") as f:
    pickle.dump(n, f)
