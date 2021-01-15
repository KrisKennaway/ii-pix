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
# - support 140px mode again
# - compare to bmp2dhr and a2bestpix
# - support LR/DLR
# - support HGR


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
    args = parser.parse_args()

    palette = palette_py.Palette()
    # screen = DHGR140Screen()
    screen = screen_py.DHGR560Screen(palette)

    image = image_py.open(screen.X_RES, screen.Y_RES, args.input)
    if args.show_input:
        Image.fromarray(image.astype(np.uint8)).show()

    dither = dither_pattern.PATTERNS[args.dither]()

    start = time.time()
    output_4bit, output_rgb = dither_pyx.dither_image(
        screen, image, dither, lookahead=args.lookahead)
    print(time.time() - start)
    screen.pack(output_4bit)

    out_image = Image.fromarray(image_py.linear_to_srgb(output_rgb).astype(
        np.uint8))
    outfile = os.path.join(os.path.splitext(args.output)[0] + ".png")
    out_image.save(outfile, "PNG")
    if args.show_output:
        out_image.show()
    # bitmap = Image.fromarray(screen.bitmap.astype('uint8') * 255)

    with open(args.output, "wb") as f:
        f.write(bytes(screen.main))
        f.write(bytes(screen.aux))


if __name__ == "__main__":
    main()
