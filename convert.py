"""Image converter to Apple II Double Hi-Res format."""

import argparse
import numpy as np


import convert_shr
import dither_pattern
import image as image_py
import palette as palette_py
import screen as screen_py


# TODO:
# - support LR/DLR
# - support HGR


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
        '--gamma-correct', type=float, default=2.4,
        help='Gamma-correct image by this value (default: 2.4)'
    )
    parser.add_argument(
        '--fixed-colours', type=int, default=0,
        help='How many colours to fix as identical across all 16 SHR palettes '
             '(default: 0)'
    )
    parser.add_argument(
        '--save-preview', action=argparse.BooleanOptionalAction, default=True,
        help='Whether to save a .PNG rendering of the output image (default: '
             'True)'
    )
    parser.add_argument(
        '--show-final-score', action=argparse.BooleanOptionalAction,
        default=False, help='Whether to output the final image quality score '
                            '(default: False)'
    )
    args = parser.parse_args()
    if args.lookahead < 1:
        parser.error('--lookahead must be at least 1')

    # palette = palette_py.PALETTES[args.palette]()
    screen = screen_py.SHR320Screen()

    # Conversion matrix from RGB to CAM16UCS colour values.  Indexed by
    # 24-bit RGB value
    rgb24_to_cam16ucs = np.load("data/rgb24_to_cam16ucs.npy")

    # Open and resize source image
    image = image_py.open(args.input)
    if args.show_input:
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        srgb_output=False).show()
    rgb = np.array(
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        gamma=args.gamma_correct)).astype(np.float32) / 255

    convert_shr.convert(screen, rgb, args)

    # dither = dither_pattern.PATTERNS[args.dither]()
    # bitmap = dither_dhr_pyx.dither_image(
    #     screen, rgb, dither, args.lookahead, args.verbose, rgb24_to_cam16ucs)

    # Show output image by rendering in target palette
    # output_palette_name = args.show_palette or args.palette
    # output_palette = palette_py.PALETTES[output_palette_name]()
    # output_screen = screen_py.DHGRScreen(output_palette)
    # if output_palette_name == "ntsc":
    #     output_srgb = output_screen.bitmap_to_image_ntsc(bitmap)
    # else:
    #     output_srgb = image_py.linear_to_srgb(
    #         output_screen.bitmap_to_image_rgb(bitmap)).astype(np.uint8)


if __name__ == "__main__":
    main()
