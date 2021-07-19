"""Image converter to Apple II Double Hi-Res format."""

import argparse
import os.path
import time

import colour
from PIL import Image
import numpy as np

import dither as dither_pyx
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
        '--resolution', type=str, choices=("140", "560"), default="560",
        help=("Effective double hi-res resolution to target.  '140' treats "
              "pixels in groups of 4, with 16 colours that are chosen "
              "independently, and ignores NTSC fringing.  This is mostly only "
              "useful for comparison to other 140px converters.  '560' treats "
              "each pixel individually, with choice of 2 colours (depending on "
              "NTSC colour phase), and looking ahead over next --lookahead "
              "pixels to optimize the colour sequence (default: 560)")
    )
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

    palette = palette_py.PALETTES[args.palette]()
    if args.resolution == "140":
        if args.palette == "ntsc":
            raise argparse.ArgumentError(
                "--resolution=140 cannot be combined with --palette=ntsc")
        screen = screen_py.DHGR140Screen(palette)
        lookahead = 0
    else:
        if args.palette == "ntsc":
            # TODO: palette depth should be controlled by Palette not Screen
            screen = screen_py.DHGR560NTSCScreen(palette)
        else:
            screen = screen_py.DHGR560Screen(palette)
        lookahead = args.lookahead

    # Open and resize source image
    image = image_py.open(args.input)
    if args.show_input:
        image_py.resize(image, screen.NATIVE_X_RES, screen.NATIVE_Y_RES * 2,
                        srgb_output=True).show()
    resized = np.array(image_py.resize(image, screen.X_RES,
                                       screen.Y_RES,
                                       gamma=args.gamma_correct)).astype(
        np.float32)

    # convert from sRGB1-linear to CAM02UCS perceptually uniform colour space
    xyz = colour.convert(
        resized / 255, "RGB", "CIE XYZ").astype(np.float32)

    # bits24 = np.arange(2**24).reshape(-1,1)
    # all_rgb = (np.concatenate(        [bits24 >> 16 & 0xff, bits24 >> 8 & 0xff, bits24 & 0xff], axis=1) /               255).astype(np.float32)
    # all_xyz = colour.convert(all_rgb, "RGB", "CIE XYZ")
    # all_cam16 = colour.convert(all_rgb, "RGB", "CAM16UCS").astype(np.float32)
    # f = np.memmap("rgb_to_cam16ucs.data", mode="w+", dtype=np.float32,
    #               shape=all_cam16.shape)
    # f[:] = all_cam16
    # if True:
    #     return

    all_cam16 = np.memmap("rgb_to_cam16ucs.data", mode="r+", dtype=np.float32,
                          shape=(2 ** 24, 3))
    dither = dither_pattern.PATTERNS[args.dither]()
    output_nbit, _ = dither_pyx.dither_image(
        screen, xyz, dither, lookahead, args.verbose, all_cam16)
    bitmap = screen.pack(output_nbit)

    # Show output image by rendering in target palette
    output_palette_name = args.show_palette or args.palette
    output_palette = palette_py.PALETTES[output_palette_name]()
    if output_palette_name == "ntsc":
        output_screen = screen_py.DHGR560NTSCScreen(output_palette)
    else:
        output_screen = screen_py.DHGR560Screen(output_palette)
    output_rgb = output_screen.bitmap_to_image_rgb(bitmap)
    out_image = Image.fromarray(image_py.linear_to_srgb(output_rgb).astype(
        np.uint8))
    out_image = image_py.resize(out_image, screen.NATIVE_X_RES,
                                screen.NATIVE_Y_RES * 2, srgb_output=True)

    if args.show_output:
        out_image.show()

    # Save Double hi-res image
    outfile = os.path.join(os.path.splitext(args.output)[0] + "-preview.png")
    out_image.save(outfile, "PNG")
    with open(args.output, "wb") as f:
        f.write(bytes(screen.aux))
        f.write(bytes(screen.main))


if __name__ == "__main__":
    main()
