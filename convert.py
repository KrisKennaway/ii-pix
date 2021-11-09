"""Image converter to Apple II Double Hi-Res format."""

import argparse
import array
import os.path
import time

import colour
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

import dither as dither_pyx
import dither_pattern
import image as image_py
import palette as palette_py
import screen as screen_py


# TODO:
# - support LR/DLR
# - support HGR


def cluster_palette(image: Image):
    colours_rgb = np.asarray(image).reshape((-1, 3))
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        colours_cam = colour.convert(colours_rgb, "RGB",
                                     "CAM16UCS").astype(np.float32)

    kmeans = KMeans(n_clusters=16, max_iter=10000)
    kmeans.fit_predict(colours_cam)
    palette_cam = kmeans.cluster_centers_
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        palette_rgb = colour.convert(palette_cam, "CAM16UCS", "RGB")
        # SHR colour palette only uses 4-bit values
        # TODO: do this more carefully
        palette_rgb = np.clip(np.round(palette_rgb * 16).astype(np.uint32) *
                              16, 0, 255)
        palette_rgb = palette_rgb.astype(np.float32) / 255
    return palette_rgb



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
                        srgb_output=True).show()
    rgb = np.array(
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        gamma=args.gamma_correct)).astype(np.float32) / 255

    palette_rgb = cluster_palette(rgb)
    output_rgb = dither_pyx.dither_shr(rgb, palette_rgb, rgb_to_cam16)
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
        srgb_output=True)

    if args.show_output:
        out_image.show()

    # Save Double hi-res image
    # outfile = os.path.join(os.path.splitext(args.output)[0] + "-preview.png")
    # out_image.save(outfile, "PNG")
    # screen.pack(bitmap)
    # with open(args.output, "wb") as f:
    #     f.write(bytes(screen.aux))
    #     f.write(bytes(screen.main))


if __name__ == "__main__":
    main()
