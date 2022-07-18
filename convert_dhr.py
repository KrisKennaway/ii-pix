import os.path

from PIL import Image
import numpy as np

import dither_dhr as dither_dhr_pyx
import dither_pattern
import palette as palette_py
import screen as screen_py
import image as image_py


def convert(screen: screen_py.DHGRScreen, rgb: np.ndarray, args):
    # Conversion matrix from RGB to CAM16UCS colour values.  Indexed by
    # 24-bit RGB value
    base_dir = os.path.dirname(__file__)
    rgb24_to_cam16ucs = np.load(
        os.path.join(base_dir, "data/rgb24_to_cam16ucs.npy"))

    dither = dither_pattern.PATTERNS[args.dither]()
    bitmap = dither_dhr_pyx.dither_image(
        screen, rgb, dither, args.lookahead, args.verbose, rgb24_to_cam16ucs)

    # Show output image by rendering in target palette
    output_palette_name = args.show_palette or args.palette
    output_palette = palette_py.PALETTES[output_palette_name]()
    output_screen = screen_py.DHGRScreen(output_palette)
    if output_palette_name == "ntsc":
        output_srgb = output_screen.bitmap_to_image_ntsc(bitmap)
    else:
        output_srgb = image_py.linear_to_srgb(
            output_screen.bitmap_to_image_rgb(bitmap)).astype(np.uint8)
    out_image = image_py.resize(
        Image.fromarray(output_srgb), screen.X_RES, screen.Y_RES * 2,
        srgb_output=True)

    if args.show_output:
        out_image.show()

    if args.save_preview:
        # Save Double hi-res image
        outfile = os.path.join(
            os.path.splitext(args.output)[0] + "-preview.png")
        out_image.save(outfile, "PNG")
    screen.pack(bitmap)
    with open(args.output, "wb") as f:
        f.write(bytes(screen.aux))
        f.write(bytes(screen.main))
