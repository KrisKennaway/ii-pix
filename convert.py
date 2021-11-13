"""Image converter to Apple II Double Hi-Res format."""

import argparse
import array
import os.path
import time
import collections
import random

import colour
from PIL import Image
import numpy as np
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import dither as dither_pyx
import dither_pattern
import image as image_py
import palette as palette_py
import screen as screen_py


# TODO:
# - support LR/DLR
# - support HGR

def cluster_palette(image: Image):
    line_to_palette = {}

    # shuffle_lines = liprint(st(range(200))
    # random.shuffle(shuffle_lines)
    # for idx, line in enumerate(shuffle_lines):
    #    line_to_palette[line] = idx % 16

    # for line in range(200):
    #     if line % 3 == 0:
    #         line_to_palette[line] = int(line / (200 / 16))
    #     elif line % 3 == 1:
    #         line_to_palette[line] = np.clip(int(line / (200 / 16)) + 1, 0, 15)
    #     else:
    #         line_to_palette[line] = np.clip(int(line / (200 / 16)) + 2, 0, 15)

    for line in range(200):
        if line % 3 == 0:
            line_to_palette[line] = int(line / (200 / 16))
        elif line % 3 == 1:
            line_to_palette[line] = np.clip(int(line / (200 / 16)) + 1, 0, 15)
        else:
            line_to_palette[line] = np.clip(int(line / (200 / 16)) + 2, 0, 15)

    colours_rgb = np.asarray(image).reshape((-1, 3))
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        colours_cam = colour.convert(colours_rgb, "RGB",
                                     "CAM16UCS").astype(np.float32)
    palettes_rgb = {}
    palettes_cam = {}
    for palette_idx in range(16):
        p_lower = max(palette_idx - 2, 0)
        p_upper = min(palette_idx + 2, 16)
        palette_pixels = colours_cam[
                         int(p_lower * (200 / 16)) * 320:int(p_upper * (
                                 200 / 16)) * 320, :]

        # kmeans = KMeans(n_clusters=16, max_iter=10000)
        # kmeans.fit_predict(palette_pixels)
        # palettes_cam[palette_idx] = kmeans.cluster_centers_

        # fixed_centroids = None
        # print(np.array(line_colours), fixed_centroids)
        # palettes_cam[palette_idx] = dither_pyx.k_means_with_fixed_centroids(
        #    16, palette_pixels, fixed_centroids=fixed_centroids,
        #    tolerance=1e-6)

        initial_centers = kmeans_plusplus_initializer(
            palette_pixels, 16).initialize()
        kmedians_instance = kmedians(palette_pixels, initial_centers)
        kmedians_instance.process()
        palettes_cam[palette_idx] = np.array(
            kmedians_instance.get_medians()).astype(np.float32)

        # palette_colours = collections.defaultdict(list)
        # for line in range(200):
        #     palette = line_to_palette[line]
        #     palette_colours[palette].extend(
        #         colours_cam[line * 320:(line + 1) * 320])

        # For each line grouping, find big palette entries with minimal total
        # distance

        # palette_cam = None
        # for palette_idx in range(16):
        #     line_colours = palette_colours[palette_idx]
        #     #if palette_idx < 15:
        #     #    line_colours += palette_colours[palette_idx + 1]
        #     # if palette_idx < 14:
        #     #     line_colours += palette_colours[palette_idx + 2]
        #     # if palette_idx > 0:
        #     #     fixed_centroids = palette_cam[:8, :]
        #     # else:
        #     fixed_centroids = None
        #     # print(np.array(line_colours), fixed_centroids)
        #     palette_cam = dither_pyx.k_means_with_fixed_centroids(16, np.array(
        #         line_colours), fixed_centroids=fixed_centroids, tolerance=1e-6)

        # kmeans = KMeans(n_clusters=16, max_iter=10000)
        # kmeans.fit_predict(line_colours)
        # palette_cam = kmeans.cluster_centers_

        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            palette_rgb = colour.convert(palettes_cam[palette_idx], "CAM16UCS",
                                         "RGB")
            # SHR colour palette only uses 4-bit values
            palette_rgb = np.round(palette_rgb * 15) / 15
            palettes_rgb[palette_idx] = palette_rgb.astype(np.float32)
    # print(palettes_rgb)

    # For each line, pick the palette with lowest total distance
    # best_palette = 15
    # for line in range(200):
    #     line_pixels = colours_cam[line*320:(line+1)*320]
    #     best_palette = dither_pyx.best_palette_for_line(
    #         line_pixels, palettes_cam, best_palette)
    #     line_to_palette[line] = best_palette
    #     print(line, line_to_palette[line])
    return palettes_cam, palettes_rgb, line_to_palette


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input image file to process.")
    parser.add_argument("output", type=str, help="Output file for converted "
                                                 "Apple II image.")
    parser.add_argument(
        "--lookahead", type=int, default=8,
        help=("How many pixels to look ahead to compensate for NTSC colour "
              "artifacts (default: 8)"))
    parser.add_argument(
        '--dither', type=str, choices=list(dither_pattern.PATTERNS.keys()),
        default=dither_pattern.DEFAULT_PATTERN,
        help="Error distribution pattern to apply when dithering (default: "
             + dither_pattern.DEFAULT_PATTERN + ")")
    parser.add_argument(
        '--show-input', action=argparse.BooleanOptionalAction, default=False,
        help="Whether to show the input image before conversion.")
    parser.add_argument(
        '--show-output', action=argparse.BooleanOptionalAction, default=True,
        help="Whether to show the output image after conversion.")
    parser.add_argument(
        '--palette', type=str, choices=list(set(palette_py.PALETTES.keys())),
        default=palette_py.DEFAULT_PALETTE,
        help='RGB colour palette to dither to.  "ntsc" blends colours over 8 '
             'pixels and gives better image quality on targets that '
             'use/emulate NTSC, but can be substantially slower.  Other '
             'palettes determine colours based on 4 pixel sequences '
             '(default: ' + palette_py.DEFAULT_PALETTE + ")")
    parser.add_argument(
        '--show-palette', type=str, choices=list(palette_py.PALETTES.keys()),
        help="RGB colour palette to use when --show_output (default: "
             "value of --palette)")
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction,
        default=False, help="Show progress during conversion")
    parser.add_argument(
        '--gamma_correct', type=float, default=2.4,
        help='Gamma-correct image by this value (default: 2.4)'
    )
    args = parser.parse_args()
    if args.lookahead < 1:
        parser.error('--lookahead must be at least 1')

    # palette = palette_py.PALETTES[args.palette]()
    screen = screen_py.SHR320Screen()

    # Conversion matrix from RGB to CAM16UCS colour values.  Indexed by
    # 24-bit RGB value
    rgb_to_cam16 = np.load("data/rgb_to_cam16ucs.npy")

    # Open and resize source image
    image = image_py.open(args.input)
    if args.show_input:
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        srgb_output=False).show()
    rgb = np.array(
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        gamma=args.gamma_correct, srgb_output=True)).astype(
        np.float32) / 255

    palettes_cam, palettes_rgb, line_to_palette = cluster_palette(rgb)
    # print(palette_rgb)
    # screen.set_palette(0, (image_py.linear_to_srgb_array(palette_rgb) *
    #                        15).astype(np.uint8))
    for i, p in palettes_rgb.items():
        screen.set_palette(i, (np.round(p * 15)).astype(np.uint8))

    output_4bit, line_to_palette = dither_pyx.dither_shr(
        rgb, palettes_cam, palettes_rgb, rgb_to_cam16)
    screen.set_pixels(output_4bit)
    output_rgb = np.zeros((200, 320, 3), dtype=np.uint8)
    for i in range(200):
        screen.line_palette[i] = line_to_palette[i]
        output_rgb[i, :, :] = (
                palettes_rgb[line_to_palette[i]][
                    output_4bit[i, :]] * 255).astype(np.uint8)
    output_srgb = image_py.linear_to_srgb(output_rgb).astype(np.uint8)

    # dither = dither_pattern.PATTERNS[args.dither]()
    # bitmap = dither_pyx.dither_image(
    #     screen, rgb, dither, args.lookahead, args.verbose, rgb_to_cam16)

    # Show output image by rendering in target palette
    # output_palette_name = args.show_palette or args.palette
    # output_palette = palette_py.PALETTES[output_palette_name]()
    # output_screen = screen_py.DHGRScreen(output_palette)
    # if output_palette_name == "ntsc":
    #     output_srgb = output_screen.bitmap_to_image_ntsc(bitmap)
    # else:
    #     output_srgb = image_py.linear_to_srgb(
    #         output_screen.bitmap_to_image_rgb(bitmap)).astype(np.uint8)
    out_image = image_py.resize(
        Image.fromarray(output_srgb), screen.X_RES, screen.Y_RES,
        srgb_output=False)  # XXX true

    if args.show_output:
        out_image.show()

    # Save Double hi-res image
    outfile = os.path.join(os.path.splitext(args.output)[0] + "-preview.png")
    out_image.save(outfile, "PNG")
    screen.pack()
    # with open(args.output, "wb") as f:
    #     f.write(bytes(screen.aux))
    #     f.write(bytes(screen.main))
    with open(args.output, "wb") as f:
        f.write(bytes(screen.memory))


if __name__ == "__main__":
    main()
