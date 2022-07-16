"""Image converter to Apple II Double Hi-Res format."""

import argparse
import numpy as np

import convert_dhr as convert_dhr_py
import convert_shr as convert_shr_py
import dither_pattern
import image as image_py
import palette as palette_py
import screen as screen_py

# TODO:
#  - support additional graphics modes (easiest --> hardest):
#    - LR/DLR
#    - SHR 3200
#    - SHR 640
#    - HGR


def add_common_args(parser):
    parser.add_argument("input", type=str, help="Input image file to process.")
    parser.add_argument("output", type=str, help="Output file for converted "
                                                 "Apple II image.")
    parser.add_argument(
        '--show-input', action=argparse.BooleanOptionalAction, default=False,
        help="Whether to show the input image before conversion.")
    parser.add_argument(
        '--show-output', action=argparse.BooleanOptionalAction, default=True,
        help="Whether to show the output image after conversion.")
    parser.add_argument(
        '--save-preview', action=argparse.BooleanOptionalAction, default=True,
        help='Whether to save a .PNG rendering of the output image (default: '
             'True)'
    )
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction,
        default=False, help="Show progress during conversion")
    parser.add_argument(
        '--gamma-correct', type=float, default=2.4,
        help='Gamma-correct image by this value (default: 2.4)'
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    dhr_parser = subparsers.add_parser("dhr")
    add_common_args(dhr_parser)

    def validate_lookahead(arg: int) -> int:
        try:
            int_arg = int(arg)
            if int_arg < 1:
                return int_arg
        except Exception:
            raise argparse.ArgumentTypeError("--lookahead must be at least 1")
    dhr_parser.add_argument(
        "--lookahead", type=validate_lookahead, default=8,
        help=("How many pixels to look ahead to compensate for NTSC colour "
              "artifacts (default: 8)"))
    dhr_parser.add_argument(
        '--dither', type=str, choices=list(dither_pattern.PATTERNS.keys()),
        default=dither_pattern.DEFAULT_PATTERN,
        help="Error distribution pattern to apply when dithering (default: "
             + dither_pattern.DEFAULT_PATTERN + ")")
    dhr_parser.add_argument(
        '--palette', type=str, choices=list(set(palette_py.PALETTES.keys())),
        default=palette_py.DEFAULT_PALETTE,
        help='RGB colour palette to dither to.  "ntsc" blends colours over 8 '
             'pixels and gives better image quality on targets that '
             'use/emulate NTSC, but can be substantially slower.  Other '
             'palettes determine colours based on 4 pixel sequences '
             '(default: ' + palette_py.DEFAULT_PALETTE + ")")
    dhr_parser.add_argument(
        '--show-palette', type=str, choices=list(palette_py.PALETTES.keys()),
        help="RGB colour palette to use when --show_output (default: "
             "value of --palette)")
    dhr_parser.set_defaults(func=convert_dhr)

    shr_parser = subparsers.add_parser("shr")
    add_common_args(shr_parser)
    shr_parser.add_argument(
        '--fixed-colours', type=int, default=0,
        help='How many colours to fix as identical across all 16 SHR palettes '
             '(default: 0)'
    )
    shr_parser.add_argument(
        '--show-final-score', action=argparse.BooleanOptionalAction,
        default=False, help='Whether to output the final image quality score '
                            '(default: False)'
    )
    shr_parser.set_defaults(func=convert_shr)
    args = parser.parse_args()
    args.func(args)


def prepare_image(image_filename: str, show_input: bool, screen,
                  gamma_correct: float) -> np.ndarray:
    # Open and resize source image
    image = image_py.open(image_filename)
    if show_input:
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        srgb_output=False).show()
    rgb = np.array(
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        gamma=gamma_correct)).astype(np.float32) / 255
    return rgb


def convert_dhr(args):
    palette = palette_py.PALETTES[args.palette]()
    screen = screen_py.DHGRScreen(palette)
    rgb = prepare_image(args.input, args.show_input, screen, args.gamma_correct)
    convert_dhr_py.convert(screen, rgb, args)


def convert_shr(args):
    screen = screen_py.SHR320Screen()
    rgb = prepare_image(args.input, args.show_input, screen, args.gamma_correct)
    convert_shr_py.convert(screen, rgb, args)


if __name__ == "__main__":
    main()
