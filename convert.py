import argparse
import os.path
import time

from PIL import Image
import numpy as np

import dither as dither_pyx
import dither_pattern
import image as image_py
import palette as palette_py
import screen as screen_py


# TODO:
# - support alternate palettes properly
# - include fringing in 140px output
# - compare to bmp2dhr and a2bestpix
# - support LR/DLR
# - support HGR
# - README


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input image file to process.")
    parser.add_argument("output", type=str, help="Output file for converted "
                                                 "Apple II image.")
    parser.add_argument(
        "--lookahead", type=int, default=6,
        help=("How many pixels to look ahead to compensate for NTSC colour "
              "artifacts."))
    parser.add_argument(
        '--dither', type=str, choices=list(dither_pattern.PATTERNS.keys()),
        default=dither_pattern.DEFAULT_PATTERN,
        help="Error distribution pattern to apply when dithering.")
    parser.add_argument(
        '--show_input', action=argparse.BooleanOptionalAction, default=False,
        help="Whether to show the input image before conversion.")
    parser.add_argument(
        '--show_output', action=argparse.BooleanOptionalAction, default=True,
        help="Whether to show the output image after conversion.")
    parser.add_argument(
        '--resolution', type=int, choices=(560, 140), default=560,
        help=("Double hi-res resolution to target.  140 treats pixels in "
              "groups of 4, with 16 colours that can be chosen independently, "
              "and ignores NTSC fringing.  560 treats each pixel individually, "
              "with choice of 2 colours (depending on NTSC colour phase), "
              "and looking ahead over next --lookahead pixels to optimize the "
              "colour sequence.")
    )
    args = parser.parse_args()

    palette = palette_py.Palette()
    if args.resolution == 560:
        screen = screen_py.DHGR560Screen(palette)
        lookahead = args.lookahead
    else:
        screen = screen_py.DHGR140Screen(palette)
        lookahead = 0

    image = image_py.open(screen.X_RES, screen.Y_RES, args.input)
    if args.show_input:
        Image.fromarray(image.astype(np.uint8)).show()

    dither = dither_pattern.PATTERNS[args.dither]()

    start = time.time()
    output_4bit, output_rgb = dither_pyx.dither_image(
        screen, image, dither, lookahead)
    print(time.time() - start)
    screen.pack(output_4bit)

    out_image = Image.fromarray(image_py.linear_to_srgb(output_rgb).astype(
        np.uint8))
    outfile = os.path.join(os.path.splitext(args.output)[0] + ".png")
    out_image.save(outfile, "PNG")
    if args.show_output:
        out_image.show()

    with open(args.output, "wb") as f:
        f.write(bytes(screen.main))
        f.write(bytes(screen.aux))


if __name__ == "__main__":
    main()