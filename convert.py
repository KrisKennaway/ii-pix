"""Image converter to Apple II Double Hi-Res format."""

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
              "artifacts.  Default: 6"))
    parser.add_argument(
        '--dither', type=str, choices=list(dither_pattern.PATTERNS.keys()),
        default=dither_pattern.DEFAULT_PATTERN,
        help="Error distribution pattern to apply when dithering.  Default: "
             + dither_pattern.DEFAULT_PATTERN)
    parser.add_argument(
        '--show_input', action=argparse.BooleanOptionalAction, default=False,
        help="Whether to show the input image before conversion.  Default: "
             "False")
    parser.add_argument(
        '--show_output', action=argparse.BooleanOptionalAction, default=True,
        help="Default: True.  Whether to show the output image after "
             "conversion.  Default: True")
    parser.add_argument(
        '--resolution', type=str, choices=("140", "560", "ntsc"), default="560",
        help=("Effective double hi-res resolution to target.  '140' treats "
              "pixels in groups of 4, with 16 colours that are chosen "
              "independently, and ignores NTSC fringing.  This is mostly only "
              "useful for comparison to other 140px converters.  '560' treats "
              "each pixel individually, with choice of 2 colours (depending on "
              "NTSC colour phase), and looking ahead over next --lookahead "
              "pixels to optimize the colour sequence.  'ntsc' additionally "
              "simulates the reduced bandwidth of the NTSC chroma signal, and "
              "causes colours to bleed over 8 successive pixels instead of 4.  "
              "Default: 560")
    )
    parser.add_argument(
        '--palette', type=str, choices=list(
            set(palette_py.PALETTES.keys()) - {"ntsc"}),
        default=palette_py.DEFAULT_PALETTE,
        help="RGB colour palette to dither to.  Ignored for "
             "--resolution=ntsc.  Default: " + palette_py.DEFAULT_PALETTE)
    parser.add_argument(
        '--show_palette', type=str, choices=list(palette_py.PALETTES.keys()),
        help="RGB colour palette to use when --show_output.  Default: "
             "value of --palette.")
    args = parser.parse_args()

    if args.resolution == "ntsc":
        palette = palette_py.PALETTES["ntsc"]()
        screen = screen_py.DHGR560NTSCScreen(palette)
        lookahead = args.lookahead
    else:
        palette = palette_py.PALETTES[args.palette]()
        if args.resolution == "560":
            screen = screen_py.DHGR560Screen(palette)
            lookahead = args.lookahead
        else:
            screen = screen_py.DHGR140Screen(palette)
            lookahead = 0

    # Open and resize source image
    image = image_py.open(args.input)
    if args.show_input:
        image_py.resize(image, 560, 384, srgb_output=True).show()
    resized = np.array(image_py.resize(image, screen.X_RES,
                                       screen.Y_RES)).astype(np.float32)

    dither = dither_pattern.PATTERNS[args.dither]()
    output_4bit, _ = dither_pyx.dither_image(
        screen, resized, dither, lookahead)
    bitmap = screen.pack(output_4bit)

    # Show output image by rendering in target palette
    if args.show_palette:
        output_palette = palette_py.PALETTES[args.show_palette]()
    else:
        output_palette = palette
    if args.show_palette == 'ntsc':
        output_screen = screen_py.DHGR560NTSCScreen(output_palette)
    else:
        output_screen = screen_py.DHGR560Screen(output_palette)
    output_rgb = output_screen.bitmap_to_image_rgb(bitmap)
    out_image = Image.fromarray(image_py.linear_to_srgb(output_rgb).astype(
        np.uint8))
    out_image = image_py.resize(out_image, 560, 384, srgb_output=True)

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
