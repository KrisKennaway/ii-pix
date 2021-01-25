"""Precompute CIE2000 perceptual colour distance matrices.

The matrix of delta-E values is computed for all pairs of 24-bit RGB values,
and 4-bit Apple II target palette.  This is a 256MB file that is mmapped at
runtime for efficient access.
"""

import argparse
import os

import image
import palette as palette_py
import colour.difference
import numpy as np

COLOURS = 256


def rgb_to_lab(rgb: np.ndarray):
    srgb = np.clip(
        image.linear_to_srgb_array(rgb.astype(np.float32) / 255), 0.0,
        1.0)
    xyz = colour.sRGB_to_XYZ(srgb)
    return colour.XYZ_to_Lab(xyz)


def all_lab_colours():
    all_rgb = np.array(tuple(np.ndindex(COLOURS, COLOURS, COLOURS)),
                       dtype=np.uint8)
    return rgb_to_lab(all_rgb)


def nearest_colours(palette, all_lab):
    diffs = np.empty((COLOURS ** 3, 16), dtype=np.float32)

    for i, palette_rgb in sorted(palette.RGB.items()):
        print("...palette colour %d" % i)
        palette_lab = rgb_to_lab(palette_rgb)
        diffs[:, i] = colour.difference.delta_E_CIE2000(all_lab, palette_lab)

    norm = np.max(diffs)
    return (diffs / norm * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--palette', type=str, choices=list(
        palette_py.PALETTES.keys()),
                        default=palette_py.DEFAULT_PALETTE,
                        help="Palette for which to compute distance matrix.")
    parser.add_argument('--all', type=bool, default=False,
                        help="Whether to compute distances for all palettes")
    args = parser.parse_args()

    if args.all:
        palette_names = list(palette_py.PALETTES.keys())
    else:
        palette_names = [args.palette]

    print("Precomputing matrix of all 24-bit LAB colours")
    all_lab = all_lab_colours()
    for palette_name in palette_names:
        print("Processing palette %s" % palette_name)
        palette = palette_py.PALETTES[palette_name](load_distances=False)
        try:
            os.mkdir(os.path.dirname(palette.DISTANCES_PATH))
        except FileExistsError:
            pass
        n = nearest_colours(palette, all_lab)
        out = np.memmap(filename=palette.DISTANCES_PATH, mode="w+",
                        dtype=np.uint8, shape=n.shape)
        out[:] = n[:]


if __name__ == "__main__":
    main()
