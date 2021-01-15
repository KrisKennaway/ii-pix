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
# - only lookahead for 560px
# - compare to bmp2dhr and a2bestpix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file to process")
    parser.add_argument("output", type=str, help="Output file for ")
    parser.add_argument(
        "--lookahead", type=int, default=6,
        help=("How many pixels to look ahead to compensate for NTSC colour "
              "artifacts."))
    parser.add_argument('--dither', type=str,
                        choices=list(dither_pattern.PATTERNS.keys()),
                        default=dither_pattern.DEFAULT_PATTERN)
    args = parser.parse_args()

    palette = palette_py.Palette()
    # screen = DHGR140Screen()
    screen = screen_py.DHGR560Screen(palette)

    image = image_py.open(screen.X_RES, screen.Y_RES, args.input)
    # image_rgb.show()

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
    out_image.show(title=outfile)
    # bitmap = Image.fromarray(screen.bitmap.astype('uint8') * 255)

    with open(args.output, "wb") as f:
        f.write(bytes(screen.main))
        f.write(bytes(screen.aux))


if __name__ == "__main__":
    main()
