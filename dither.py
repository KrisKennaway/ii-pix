import argparse
import functools
import os.path
from typing import Tuple

from PIL import Image
import colour.difference
import numpy as np


# TODO:
# - precompute lab differences
# - only lookahead for 560px
# - palette class
# - compare to bmp2dhr and a2bestpix

def srgb_to_linear_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** gamma)


def linear_to_srgb_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1.0 / gamma) -
                    0.055)


# XXX work uniformly with 255 or 1.0 range
def srgb_to_linear(im: np.ndarray) -> np.ndarray:
    rgb_linear = srgb_to_linear_array(im / 255.0, gamma=2.4)
    return (np.clip(rgb_linear, 0.0, 1.0) * 255).astype(np.float32)


def linear_to_srgb(im: np.ndarray) -> np.ndarray:
    srgb = linear_to_srgb_array(im / 255.0, gamma=2.4)
    return (np.clip(srgb, 0.0, 1.0) * 255).astype(np.float32)


# Default bmp2dhr palette
RGB = {
    (False, False, False, False): np.array((0, 0, 0)),  # Black
    (False, False, False, True): np.array((148, 12, 125)),  # Magenta
    (False, False, True, False): np.array((99, 77, 0)),  # Brown
    (False, False, True, True): np.array((249, 86, 29)),  # Orange
    (False, True, False, False): np.array((51, 111, 0)),  # Dark green
    # XXX RGB values are used as keys in DOTS dict, need to be unique
    (False, True, False, True): np.array((126, 126, 125)),  # Grey1
    (False, True, True, False): np.array((67, 200, 0)),  # Green
    (False, True, True, True): np.array((221, 206, 23)),  # Yellow
    (True, False, False, False): np.array((32, 54, 212)),  # Dark blue
    (True, False, False, True): np.array((188, 55, 255)),  # Violet
    (True, False, True, False): np.array((126, 126, 126)),  # Grey2
    (True, False, True, True): np.array((255, 129, 236)),  # Pink
    (True, True, False, False): np.array((7, 168, 225)),  # Med blue
    (True, True, False, True): np.array((158, 172, 255)),  # Light blue
    (True, True, True, False): np.array((93, 248, 133)),  # Aqua
    (True, True, True, True): np.array((255, 255, 255)),  # White
}

# OpenEmulator
sRGB = {
    (False, False, False, False): np.array((0, 0, 0)),  # Black
    (False, False, False, True): np.array((206, 0, 123)),  # Magenta
    (False, False, True, False): np.array((100, 105, 0)),  # Brown
    (False, False, True, True): np.array((247, 79, 0)),  # Orange
    (False, True, False, False): np.array((0, 153, 0)),  # Dark green
    # XXX RGB values are used as keys in DOTS dict, need to be unique
    (False, True, False, True): np.array((131, 132, 132)),  # Grey1
    (False, True, True, False): np.array((0, 242, 0)),  # Green
    (False, True, True, True): np.array((216, 220, 0)),  # Yellow
    (True, False, False, False): np.array((21, 0, 248)),  # Dark blue
    (True, False, False, True): np.array((235, 0, 242)),  # Violet
    (True, False, True, False): np.array((140, 140, 140)),  # Grey2  # XXX
    (True, False, True, True): np.array((244, 104, 240)),  # Pink
    (True, True, False, False): np.array((0, 181, 248)),  # Med blue
    (True, True, False, True): np.array((160, 156, 249)),  # Light blue
    (True, True, True, False): np.array((21, 241, 132)),  # Aqua
    (True, True, True, True): np.array((244, 247, 244)),  # White
}

# # Virtual II (sRGB)
# sRGB = {
#     (False, False, False, False): np.array((0, 0, 0)),  # Black
#     (False, False, False, True): np.array((231,36,66)),  # Magenta
#     (False, False, True, False): np.array((154,104,0)),  # Brown
#     (False, False, True, True): np.array((255,124,0)),  # Orange
#     (False, True, False, False): np.array((0,135,45)),  # Dark green
#     (False, True, False, True): np.array((104,104,104)),  # Grey2 XXX
#     (False, True, True, False): np.array((0,222,0)),  # Green
#     (False, True, True, True): np.array((255,252,0)),  # Yellow
#     (True, False, False, False): np.array((1,30,169)),  # Dark blue
#     (True, False, False, True): np.array((230,73,228)),  # Violet
#     (True, False, True, False): np.array((185,185,185)),  # Grey1 XXX
#     (True, False, True, True): np.array((255,171,153)),  # Pink
#     (True, True, False, False): np.array((47,69,255)),  # Med blue
#     (True, True, False, True): np.array((120,187,255)),  # Light blue
#     (True, True, True, False): np.array((83,250,208)),  # Aqua
#     (True, True, True, True): np.array((255, 255, 255)),  # White
# }
RGB = {}
for k, v in sRGB.items():
    RGB[k] = (np.clip(srgb_to_linear_array(v / 255), 0.0, 1.0) * 255).astype(
        np.uint8)


class ColourDistance:
    @staticmethod
    def distance(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        raise NotImplementedError


class RGBDistance(ColourDistance):
    """Euclidean squared distance in RGB colour space."""

    @staticmethod
    def distance(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        return float(np.asscalar(np.sum(np.power(np.array(rgb1) -
                                                 np.array(rgb2), 2))))


def rgb_to_lab(rgb: np.ndarray):
    srgb = np.clip(
        linear_to_srgb_array(np.array(rgb, dtype=np.float32) / 255), 0.0,
        1.0)
    xyz = colour.sRGB_to_XYZ(srgb)
    return colour.XYZ_to_Lab(xyz)


LAB = {}
for k, v in RGB.items():
    LAB[k] = rgb_to_lab(v)

DOTS = {}
for k, v in RGB.items():
    DOTS[tuple(v)] = k


class CIE2000Distance(ColourDistance):
    """CIE2000 delta-E distance."""

    def _nearest_colours(self):
        diffs = np.empty_like((256 ** 3, 16), dtype=np.float32)

        all_rgb = np.array(tuple(np.ndindex(256, 256, 256)),
                           dtype=np.uint8)
        all_srgb = linear_to_srgb(all_rgb / 255) * 255
        all_xyz = colour.sRGB_to_XYZ(all_srgb)
        all_lab = colour.XYZ_to_Lab(all_xyz)

        for i, p in enumerate(LAB.values()):
            diffs[:, i] = colour.difference.delta_E_CIE2000(all_lab, p)
        self.diffs = diffs

    @staticmethod
    def distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
        return colour.difference.delta_E_CIE2000(lab1, lab2)


class LABEuclideanDistance(ColourDistance):
    """Euclidean distance in LAB colour space."""

    @staticmethod
    def distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
        return np.sqrt(np.sum(np.power(lab1 - lab2, 2), axis=2))


# class CCIR601Distance(ColourDistance):
#     @staticmethod
#     def _to_luma(rgb: np.ndarray):
#         return rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
#
#     def distance(self, rgb1: np.ndarray, rgb2: np.ndarray) -> float:
#         delta_rgb = ((rgb1[0] - rgb2[0]) / 255, (rgb1[1] - rgb2[1]) / 255,
#                      (rgb1[2] - rgb2[2]) / 255)
#         luma_diff = (self._to_luma(rgb1) - self._to_luma(rgb2)) / 255
#
#         # TODO: this is the formula bmp2dhr uses but what motivates it?
#         return (
#                        delta_rgb[0] * delta_rgb[0] * 0.299 +
#                        delta_rgb[1] * delta_rgb[1] * 0.587 +
#                        delta_rgb[2] * delta_rgb[2] * 0.114) * 0.75 + (
#                        luma_diff * luma_diff)


class Screen:
    X_RES = None
    Y_RES = None
    X_PIXEL_WIDTH = None

    def __init__(self):
        self.main = np.zeros(8192, dtype=np.uint8)
        self.aux = np.zeros(8192, dtype=np.uint8)

    @staticmethod
    def y_to_base_addr(y: int) -> int:
        """Maps y coordinate to screen memory base address."""
        a = y // 64
        d = y - 64 * a
        b = d // 8
        c = d - 8 * b

        return 1024 * c + 128 * b + 40 * a

    def _image_to_bitmap(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pack(self, image: np.ndarray):
        bitmap = self._image_to_bitmap(image)
        # The DHGR display encodes 7 pixels across interleaved 4-byte sequences
        # of AUX and MAIN memory, as follows:
        # PBBBAAAA PDDCCCCB PFEEEEDD PGGGGFFF
        # Aux N    Main N   Aux N+1  Main N+1  (N even)
        main_col = np.zeros(
            (self.Y_RES, self.X_RES * self.X_PIXEL_WIDTH // 14), dtype=np.uint8)
        aux_col = np.zeros(
            (self.Y_RES, self.X_RES * self.X_PIXEL_WIDTH // 14), dtype=np.uint8)
        for byte_offset in range(80):
            column = np.zeros(self.Y_RES, dtype=np.uint8)
            for bit in range(7):
                column |= (bitmap[:, 7 * byte_offset + bit].astype(
                    np.uint8) << bit)
            if byte_offset % 2 == 0:
                aux_col[:, byte_offset // 2] = column
            else:
                main_col[:, (byte_offset - 1) // 2] = column

        for y in range(self.Y_RES):
            addr = self.y_to_base_addr(y)
            self.aux[addr:addr + 40] = aux_col[y, :]
            self.main[addr:addr + 40] = main_col[y, :]

    @staticmethod
    def pixel_palette_options(last_pixel, x: int):
        raise NotImplementedError

    @staticmethod
    def find_closest_color(
            pixel, palette_options, palette_options_lab, differ:
            ColourDistance):
        best = np.argmin(differ.distance(pixel, palette_options_lab))
        return palette_options[best]


class DHGR140Screen(Screen):
    """DHGR screen ignoring colour fringing, i.e. treating as 140x192x16."""

    X_RES = 140
    Y_RES = 192
    X_PIXEL_WIDTH = 4

    def _image_to_bitmap(self, image_rgb: np.ndarray) -> np.ndarray:
        bitmap = np.zeros(
            (self.Y_RES, self.X_RES * self.X_PIXEL_WIDTH), dtype=np.bool)
        for y in range(self.Y_RES):
            for x in range(self.X_RES):
                pixel = image_rgb[y, x]
                dots = DOTS[pixel]
                bitmap[y, x * self.X_PIXEL_WIDTH:(
                        (x + 1) * self.X_PIXEL_WIDTH)] = dots
        return bitmap

    @staticmethod
    def pixel_palette_options(last_pixel, x: int):
        return np.array(list(RGB.values())), np.array(list(LAB.values()))


class DHGR560Screen(Screen):
    """DHGR screen including colour fringing."""
    X_RES = 560
    Y_RES = 192
    X_PIXEL_WIDTH = 1

    def _image_to_bitmap(self, image: np.ndarray) -> np.ndarray:
        bitmap = np.zeros((self.Y_RES, self.X_RES), dtype=np.bool)
        for y in range(self.Y_RES):
            for x in range(self.X_RES):
                pixel = image[y, x]
                dots = DOTS[tuple(pixel)]
                phase = x % 4
                bitmap[y, x] = dots[phase]
        return bitmap

    @staticmethod
    def pixel_palette_options(last_pixel_rgb, x: int):
        last_dots = DOTS[tuple(last_pixel_rgb)]
        other_dots = list(last_dots)
        other_dots[x % 4] = not other_dots[x % 4]
        other_dots = tuple(other_dots)
        return (
            np.array([RGB[last_dots], RGB[other_dots]]),
            np.array([LAB[last_dots], LAB[other_dots]]))


class Dither:
    PATTERN = None
    ORIGIN = None

    def dither_bounds(self, screen: Screen, x: int, y: int):
        pshape = self.PATTERN.shape
        et = max(self.ORIGIN[0] - y, 0)
        eb = min(pshape[0], screen.Y_RES - 1 - y)
        el = max(self.ORIGIN[1] - x, 0)
        er = min(pshape[1], screen.X_RES - 1 - x)

        yt = y - self.ORIGIN[0] + et
        yb = y - self.ORIGIN[0] + eb
        xl = x - self.ORIGIN[1] + el
        xr = x - self.ORIGIN[1] + er

        return et, eb, el, er, yt, yb, xl, xr

    def apply(self, screen: Screen, image: np.ndarray, x: int, y: int,
              quant_error: np.ndarray, one_line=False):
        pshape = self.PATTERN.shape
        error = self.PATTERN.reshape(
            (pshape[0], pshape[1], 1)) * quant_error.reshape((1, 1, 3))
        et, eb, el, er, yt, yb, xl, xr = self.dither_bounds(screen, x, y)
        if one_line:
            yb = yt + 1
            eb = et + 1
        # TODO: compare without clipping here, i.e. allow RGB values to exceed
        # 0-255 range
        image[yt:yb, xl:xr, :] = np.clip(
            image[yt:yb, xl:xr, :] + error[et:eb, el:er, :], 0, 255)


class FloydSteinbergDither(Dither):
    # 0 * 7
    # 3 5 1
    PATTERN = np.array(((0, 0, 7), (3, 5, 1))) / 16
    ORIGIN = (0, 1)


class BuckelsDither(Dither):
    # 0 * 2 1
    # 1 2 1 0
    # 0 1 0 0
    PATTERN = np.array(((0, 0, 2, 1), (1, 2, 1, 0), (0, 1, 0, 0))) / 8
    ORIGIN = (0, 1)


class JarvisDither(Dither):
    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array(((0, 0, 0, 7, 5), (3, 5, 7, 5, 3), (1, 3, 5, 3, 1))) / 48
    ORIGIN = (0, 2)


# XXX needed?
def SRGBResize(im, size, filter) -> np.ndarray:
    # Convert to numpy array of float
    arr = np.array(im, dtype=np.float32) / 255.0
    # Convert sRGB -> linear
    arr = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    # Resize using PIL
    arrOut = np.zeros((size[1], size[0], arr.shape[2]))
    for i in range(arr.shape[2]):
        chan = Image.fromarray(arr[:, :, i])
        chan = chan.resize(size, filter)
        arrOut[:, :, i] = np.array(chan).clip(0.0, 1.0)
    # Convert linear -> sRGB
    arrOut = np.where(arrOut <= 0.0031308, 12.92 * arrOut,
                      1.055 * arrOut ** (1.0 / 2.4) - 0.055)
    arrOut = np.rint(np.clip(arrOut, 0.0, 1.0) * 255.0)
    return arrOut


def open_image(screen: Screen, filename: str) -> np.ndarray:
    im = Image.open(filename)
    # TODO: convert to sRGB colour profile explicitly, in case it has some other
    #  profile already.
    if im.mode != "RGB":
        im = im.convert("RGB")
    return srgb_to_linear(
        SRGBResize(im, (screen.X_RES, screen.Y_RES),
                   Image.LANCZOS))


@functools.lru_cache(None)
def lookahead_options(screen, lookahead, last_pixel_rgb, x):
    options_rgb = np.empty((2 ** lookahead, lookahead, 3), dtype=np.float32)
    options_lab = np.empty((2 ** lookahead, lookahead, 3), dtype=np.float32)
    for i in range(2 ** lookahead):
        output_pixel_rgb = np.array(last_pixel_rgb)
        for j in range(lookahead):
            xx = x + j
            palette_choices, palette_choices_lab = screen.pixel_palette_options(
                output_pixel_rgb, xx)
            output_pixel_lab = np.array(
                palette_choices_lab[(i & (1 << j)) >> j])
            output_pixel_rgb = np.array(
                palette_choices[(i & (1 << j)) >> j])
            # XXX copy
            options_lab[i, j, :] = np.copy(output_pixel_lab)
            options_rgb[i, j, :] = np.copy(output_pixel_rgb)

    return options_rgb, options_lab


def ideal_dither(screen: Screen, image: np.ndarray, image_lab: np.ndarray,
                 dither: Dither, differ: ColourDistance, x, y,
                 lookahead) -> np.ndarray:
    et, eb, el, er, yt, yb, xl, xr = dither.dither_bounds(screen, x, y)
    # XXX tighten bounding box
    ideal_dither = np.empty_like(image)
    ideal_dither[yt:yb, :, :] = np.copy(image[yt:yb, :, :])

    ideal_dither_lab = np.empty_like(image_lab)
    ideal_dither_lab[yt:yb, :, :] = np.copy(image_lab[yt:yb, :, :])

    palette_choices = np.array(list(RGB.values()))
    palette_choices_lab = np.array(list(LAB.values()))
    for xx in range(x, min(max(x + lookahead, xr), screen.X_RES)):
        input_pixel = np.copy(ideal_dither[y, xx, :])
        input_pixel_lab = rgb_to_lab(np.clip(input_pixel), 0, 255)
        ideal_dither_lab[y, xx, :] = input_pixel_lab
        output_pixel = screen.find_closest_color(input_pixel_lab,
                                                 palette_choices,
                                                 palette_choices_lab,
                                                 differ)
        quant_error = input_pixel - output_pixel
        ideal_dither[y, xx, :] = output_pixel
        # XXX don't care about other y values
        dither.apply(screen, ideal_dither, xx, y, quant_error)

    return ideal_dither_lab


def dither_lookahead(
        screen: Screen, image_rgb: np.ndarray, image_lab: np.ndarray,
        dither: Dither, differ: ColourDistance, x, y, last_pixel_rgb,
        lookahead) -> np.ndarray:
    et, eb, el, er, yt, yb, xl, xr = dither.dither_bounds(screen, x, y)

    # X coord value of larger of dither bounding box or lookahead horizon
    xxr = min(max(x + lookahead, xr), screen.X_RES)

    # copies of input pixels so we can dither in bulk
    # Leave enough space so we can dither the last of our lookahead pixels
    lah_image_rgb = np.zeros(
        (2 ** lookahead, lookahead + xr - xl, 3), dtype=np.float32)
    lah_image_rgb[:, 0:xxr - x, :] = image_rgb[y, x:xxr, :]

    options_rgb, options_lab = lookahead_options(
        screen, lookahead, tuple(last_pixel_rgb), x % 4)
    for i in range(xxr - x):
        # options_rgb choices are fixed, but we can still distribute
        # quantization error from having made these choices, in order to compute
        # the total error
        input_pixels = lah_image_rgb[:, i, :]
        output_pixels = options_rgb[:, i, :]
        quant_error = input_pixels - output_pixels
        # Don't update the input at position x (since we've already chosen
        # deterministic outputs), but do propagate quantization
        # errors to positions >x so we can compensate for how good/bad these
        # choices were
        # XXX vectorize
        for j in range(2 ** lookahead):
            # print(quant_error[j])
            dither.apply(
                screen, lah_image_rgb[j, :, :].reshape(1, -1, 3),
                i, 0, quant_error[j], one_line=True)

    # print("options=", options_rgb)
    # print("rgb=",lah_image_rgb)
    lah_image_lab = rgb_to_lab(np.clip(lah_image_rgb[:, 0:lookahead, :], 0,
                                       255))
    error = differ.distance(lah_image_lab, options_lab)
    # print(lah_image_lab)
    # print("error=", error)
    total_error = np.sum(np.power(error, 2), axis=1)
    # print("total_error=",total_error)
    best = np.argmin(total_error)
    # print("best=",best)
    return options_rgb[best, 0, :], options_lab[best, 0, :]


def dither_image(
        screen: Screen, image_rgb: np.ndarray, dither: Dither, differ:
        ColourDistance, lookahead) -> np.ndarray:
    image_lab = rgb_to_lab(image_rgb)

    for y in range(screen.Y_RES):
        print(y)
        output_pixel_rgb = np.array((0, 0, 0), dtype=np.float32)
        for x in range(screen.X_RES):
            input_pixel_rgb = image_rgb[y, x, :]
            # Make sure lookahead region is updated from previously applied
            # dithering
            et, eb, el, er, yt, yb, xl, xr = dither.dither_bounds(screen, x, y)
            image_lab[y, x:xr, :] = rgb_to_lab(
                np.clip(image_rgb[y, x:xr, :], 0, 255))

            # ideal_lab = ideal_dither(screen, image_rgb, image_lab, dither,
            #                         differ, x, y, lookahead)
            output_pixel_rgb, output_pixel_lab = dither_lookahead(
                screen, image_rgb, image_lab, dither, differ, x, y,
                output_pixel_rgb, lookahead)
            # print(output_pixel_rgb, output_pixel_lab)
            quant_error = input_pixel_rgb - output_pixel_rgb
            image_rgb[y, x, :] = output_pixel_rgb
            dither.apply(screen, image_rgb, x, y, quant_error)

        # if y == 1:
        #    return
    return image_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file to process")
    parser.add_argument("output", type=str, help="Output file for ")
    parser.add_argument(
        "--lookahead", type=int, default=4,
        help=("How many pixels to look ahead to compensate for NTSC colour "
              "artifacts."))
    args = parser.parse_args()

    # screen = DHGR140Screen()
    screen = DHGR560Screen()

    image = open_image(screen, args.input)
    # image.show()

    # dither = FloydSteinbergDither()
    # dither = BuckelsDither()
    dither = JarvisDither()

    differ = CIE2000Distance()
    # differ = LABEuclideanDistance()
    # differ = CCIR601Distance()

    output = dither_image(screen, image, dither, differ,
                          lookahead=args.lookahead)
    screen.pack(output)

    out_image = Image.fromarray(linear_to_srgb(output).astype(np.uint8))
    outfile = os.path.join(os.path.splitext(args.output)[0] + ".png")
    out_image.save(outfile, "PNG")
    out_image.show(title=outfile)
    # bitmap = Image.fromarray(screen.bitmap.astype('uint8') * 255)

    with open(args.output, "wb") as f:
        f.write(bytes(screen.main))
        f.write(bytes(screen.aux))


if __name__ == "__main__":
    main()
