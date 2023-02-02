import os.path

from PIL import Image
import numpy as np

import dither_dhr as dither_dhr_pyx
import dither_pattern
import palette as palette_py
import screen as screen_py
import image as image_py


def _output(out_image: Image, args):
    if args.show_output:
        out_image.show()

    if args.save_preview:
        # Save Hi-res image
        outfile = os.path.join(
            os.path.splitext(args.output)[0] + "-preview.png")
        out_image.save(outfile, "PNG")


def _write(screen: screen_py.HGRScreen, linear_bytemap: np.ndarray, args):
    screen.pack_bytes(linear_bytemap)
    with open(args.output, "wb") as f:
        f.write(bytes(screen.main))


# TODO: unify with convert_dhr.convert()
def convert(screen: screen_py.HGRScreen, image: Image, args):
    rgb = np.array(image).astype(np.float32) / 255

    # Conversion matrix from RGB to CAM16UCS colour values.  Indexed by
    # 24-bit RGB value
    base_dir = os.path.dirname(__file__)
    rgb24_to_cam16ucs = np.load(
        os.path.join(base_dir, "data/rgb24_to_cam16ucs.npy"))

    dither = dither_pattern.PATTERNS[args.dither](
        error_fraction = args.error_fraction)
    bitmap, linear_bytemap = dither_dhr_pyx.dither_image(
        screen, rgb, dither, 8, args.verbose, rgb24_to_cam16ucs)

    # Show output image by rendering in target palette
    output_palette_name = args.show_palette or args.palette
    output_palette = palette_py.PALETTES[output_palette_name]()
    output_screen = screen_py.HGRScreen(output_palette)
    if output_palette_name == "ntsc":
        output_srgb = output_screen.bitmap_to_image_ntsc(bitmap)
    else:
        output_srgb = image_py.linear_to_srgb(
            output_screen.bitmap_to_image_rgb(bitmap)).astype(np.uint8)
    out_image = image_py.resize(
        Image.fromarray(output_srgb), screen.X_RES, screen.Y_RES * 2,
        srgb_output=True)

    _output(out_image, args)
    _write(screen, linear_bytemap, args)
