"""Precompute CIE2000 perceptual colour distance matrices.

The matrix of delta-E values is computed for all pairs of 24-bit RGB values,
and Apple II target palette values.  This is written out as a file that is
mmapped at runtime for efficient access.  For a 16-colour target palette this
file is 256MB; for a 256-colour (NTSC) target palette it is 4GB.
"""

import argparse
import os

import image
import palette as palette_py
import colour.difference
import numpy as np

RGB_LEVELS = 256
# Largest possible value of delta_E_CIE2000 between two RGB values
DELTA_E_MAX = 120  # TODO: fine-tune


def rgb_to_lab(rgb: np.ndarray):
    srgb = np.clip(
        image.linear_to_srgb_array(rgb.astype(np.float32) / 255), 0.0,
        1.0)
    xyz = colour.sRGB_to_XYZ(srgb)
    return colour.XYZ_to_Lab(xyz)


def all_lab_colours():
    all_rgb = np.array(tuple(np.ndindex(RGB_LEVELS, RGB_LEVELS, RGB_LEVELS)),
                       dtype=np.uint8)
    return rgb_to_lab(all_rgb)


def nearest_colours(palette, all_lab, diffs):
    palette_size = len(palette.RGB)
    palette_labs = np.empty((palette_size, 3), dtype=np.float32)
    for i, palette_rgb in palette.RGB.items():
        palette_labs[i, :] = rgb_to_lab(palette_rgb)

    print("Computing all 24-bit palette diffs:")
    for i in range(palette_size):
        print("  %d/%d" % (i, palette_size))
        # Compute all palette diffs for a block of 65536 successive RGB
        # source values at once, which bounds the memory use while also writing
        # contiguously to the mmapped array.
        diffs[i * (1 << 16):(i + 1) * (1 << 16), :] = (
                colour.difference.delta_E_CIE2000(
                    all_lab[i * (1 << 16):(i + 1) * (
                            1 << 16)].reshape((1 << 16, 1, 3)),
                    palette_labs.reshape((1, palette_size, 3))) / DELTA_E_MAX *
                255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--palette', type=str, choices=list(
        palette_py.PALETTES.keys()),
                        default=palette_py.DEFAULT_PALETTE,
                        help="Palette for which to compute distance matrix.")
    parser.add_argument('--all', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Whether to compute distances for all palettes")
    args = parser.parse_args()

    if args.all:
        palette_names = list(palette_py.PALETTES.keys())
    else:
        palette_names = [args.palette]

    print("Precomputing matrix of all 24-bit LAB colours")
    all_lab = all_lab_colours()
    for palette_name in palette_names:
        print("Creating distance file for palette %s" % palette_name)
        palette = palette_py.PALETTES[palette_name](load_distances=False)
        try:
            os.mkdir(os.path.dirname(palette.DISTANCES_PATH))
        except FileExistsError:
            pass
        out = np.memmap(filename=palette.DISTANCES_PATH, mode="w+",
                        dtype=np.uint8, shape=(RGB_LEVELS ** 3,
                                               len(palette.RGB)))
        nearest_colours(palette, all_lab, out)


if __name__ == "__main__":
    main()
