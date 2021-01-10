import argparse
import bz2
import functools
import os.path
import pickle
from typing import Tuple

from PIL import Image
import numpy as np
import pyximport; pyximport.install(language_level=3)
import dither_apply

# TODO:
# - only lookahead for 560px
# - palette class
# - compare to bmp2dhr and a2bestpix

def srgb_to_linear_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** gamma)


def linear_to_srgb_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1.0 / gamma) -
                    0.055)


def srgb_to_linear(im: np.ndarray) -> np.ndarray:
    rgb_linear = srgb_to_linear_array(im / 255.0, gamma=2.4)
    return (np.clip(rgb_linear, 0.0, 1.0) * 255).astype(np.float32)


def linear_to_srgb(im: np.ndarray) -> np.ndarray:
    srgb = linear_to_srgb_array(im / 255.0, gamma=2.4)
    return (np.clip(srgb, 0.0, 1.0) * 255).astype(np.float32)


# Default bmp2dhr palette
RGB = {
    0: np.array((0, 0, 0)),  # Black
    8: np.array((148, 12, 125)),  # Magenta
    4: np.array((99, 77, 0)),  # Brown
    12: np.array((249, 86, 29)),  # Orange
    2: np.array((51, 111, 0)),  # Dark green
    10: np.array((126, 126, 125)),  # Grey2
    6: np.array((67, 200, 0)),  # Green
    14: np.array((221, 206, 23)),  # Yellow
    1: np.array((32, 54, 212)),  # Dark blue
    9: np.array((188, 55, 255)),  # Violet
    5: np.array((126, 126, 126)),  # Grey1
    13: np.array((255, 129, 236)),  # Pink
    3: np.array((7, 168, 225)),  # Med blue
    11: np.array((158, 172, 255)),  # Light blue
    7: np.array((93, 248, 133)),  # Aqua
    15: np.array((255, 255, 255)),  # White
}

# Maps palette values to screen dots.  Note that these are the same as
# the binary values in reverse order.
DOTS = {
    0: (False, False, False, False),
    1: (True, False, False, False),
    2: (False, True, False, False),
    3: (True, True, False, False),
    4: (False, False, True, False),
    5: (True, False, True, False),
    6: (False, True, True, False),
    7: (True, True, True, False),
    8: (False, False, False, True),
    9: (True, False, False, True),
    10: (False, True, False, True),
    11: (True, True, False, True),
    12: (False, False, True, True),
    13: (True, False, True, True),
    14: (False, True, True, True),
    15: (True, True, True, True)
}
DOTS_TO_4BIT = {}
for k, v in DOTS.items():
    DOTS_TO_4BIT[v] = k

# OpenEmulator
sRGB = {
    0: np.array((0, 0, 0)),  # Black
    8: np.array((206, 0, 123)),  # Magenta
    4: np.array((100, 105, 0)),  # Brown
    12: np.array((247, 79, 0)),  # Orange
    2: np.array((0, 153, 0)),  # Dark green
    # XXX RGB values are used as keys in DOTS dict, need to be unique
    10: np.array((131, 132, 132)),  # Grey2
    6: np.array((0, 242, 0)),  # Green
    14: np.array((216, 220, 0)),  # Yellow
    1: np.array((21, 0, 248)),  # Dark blue
    9: np.array((235, 0, 242)),  # Violet
    5: np.array((140, 140, 140)),  # Grey1  # XXX
    13: np.array((244, 104, 240)),  # Pink
    3: np.array((0, 181, 248)),  # Med blue
    11: np.array((160, 156, 249)),  # Light blue
    7: np.array((21, 241, 132)),  # Aqua
    15: np.array((244, 247, 244)),  # White
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
    def distance(rgb1: np.ndarray, rgb2: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CIE2000Distance(ColourDistance):
    """CIE2000 delta-E distance."""

    def __init__(self):
        with bz2.open("nearest.pickle.bz2", "rb") as f:
            self._distances = pickle.load(f)
            assert self._distances.dtype == np.uint8

    @staticmethod
    def _flatten_rgb(rgb):
        return (rgb[..., 0] << 16) + (rgb[..., 1] << 8) + (rgb[..., 2])

    def distance(self, rgb: np.ndarray, bit4: np.ndarray) -> np.ndarray:
        rgb24 = self._flatten_rgb(np.clip(rgb, 0, 255).astype(np.int))
        return self._distances[rgb24, bit4].astype(np.int)


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


class DHGR140Screen(Screen):
    """DHGR screen ignoring colour fringing, i.e. treating as 140x192x16."""

    X_RES = 140
    Y_RES = 192
    X_PIXEL_WIDTH = 4

    def _image_to_bitmap(self, image_4bit: np.ndarray) -> np.ndarray:
        bitmap = np.zeros(
            (self.Y_RES, self.X_RES * self.X_PIXEL_WIDTH), dtype=np.bool)
        for y in range(self.Y_RES):
            for x in range(self.X_RES):
                pixel = image_4bit[y, x].item()
                dots = DOTS[pixel]
                bitmap[y, x * self.X_PIXEL_WIDTH:(
                        (x + 1) * self.X_PIXEL_WIDTH)] = dots
        return bitmap

    @staticmethod
    def pixel_palette_options(last_pixel_4bit, x: int):
        return np.array(list(RGB.keys())), np.array(list(RGB.values()))


class DHGR560Screen(Screen):
    """DHGR screen including colour fringing."""
    X_RES = 560
    Y_RES = 192
    X_PIXEL_WIDTH = 1

    def _image_to_bitmap(self, image_4bit: np.ndarray) -> np.ndarray:
        bitmap = np.zeros((self.Y_RES, self.X_RES), dtype=np.bool)
        for y in range(self.Y_RES):
            for x in range(self.X_RES):
                pixel = image_4bit[y, x].item()
                dots = DOTS[pixel]
                phase = x % 4
                bitmap[y, x] = dots[phase]
        return bitmap

    @staticmethod
    def pixel_palette_options(last_pixel_4bit, x: int):
        last_dots = DOTS[last_pixel_4bit]
        other_dots = list(last_dots)
        other_dots[x % 4] = not other_dots[x % 4]
        other_dots = tuple(other_dots)
        other_pixel_4bit = DOTS_TO_4BIT[other_dots]
        return (
            np.array([last_pixel_4bit, other_pixel_4bit]),
            np.array([RGB[last_pixel_4bit], RGB[other_pixel_4bit]]))


class Dither:
    PATTERN = None
    ORIGIN = None

    @functools.lru_cache(None)
    def x_dither_bounds(self, screen: Screen, x: int):
        pshape = self.PATTERN.shape
        el = max(self.ORIGIN[1] - x, 0)
        er = min(pshape[1], screen.X_RES - 1 - x)

        xl = x - self.ORIGIN[1] + el
        xr = x - self.ORIGIN[1] + er

        return el, er, xl, xr

    @functools.lru_cache(None)
    def y_dither_bounds(self, screen: Screen, y: int, one_line=False):
        pshape = self.PATTERN.shape
        et = max(self.ORIGIN[0] - y, 0)
        eb = min(pshape[0], screen.Y_RES - 1 - y)

        yt = y - self.ORIGIN[0] + et
        yb = y - self.ORIGIN[0] + eb

        if one_line:
            yb = yt + 1
            eb = et + 1

        return et, eb, yt, yb

    def apply(self, screen: Screen, image: np.ndarray, x: int, y: int,
              quant_error: np.ndarray, one_line=False):
        el, er, xl, xr = self.x_dither_bounds(screen, x)
        et, eb, yt, yb = self.y_dither_bounds(screen, y, one_line)
        return dither_apply.apply(self.PATTERN, el, er, xl, xr, et, eb, yt,
                                  yb, image, quant_error)
        # error = self.PATTERN * quant_error.reshape((1, 1, 3))
        #
        # # We could avoid clipping here, i.e. allow RGB values to extend beyond
        # # 0..255 to capture a larger range of residual error.  This is faster
        # # but seems to reduce image quality.
        # # XXX extend image region to avoid need for boundary box clipping
        # image[yt:yb, xl:xr, :] = np.clip(
        #     image[yt:yb, xl:xr, :] + error[et:eb, el:er, :], 0, 255)

    def apply_one_line(self, screen: Screen, image: np.ndarray, x: int, y: int,
                       quant_error: np.ndarray):
        el, er, xl, xr = self.x_dither_bounds(screen, x)
        return dither_apply.apply_one_line(self.PATTERN, el, er, xl, xr, y,
                                           image, quant_error)
        # error = self.PATTERN[0, :] * quant_error.reshape(1, 3)
        #
        # image[y, xl:xr, :] = np.clip(
        #     image[y, xl:xr, :] + error[el:er, :], 0, 255)


class FloydSteinbergDither(Dither):
    # 0 * 7
    # 3 5 1
    PATTERN = np.array(((0, 0, 7), (3, 5, 1)),
                       dtype=np.float32).reshape(2, 3, 1) / np.float(16)
    # XXX X_ORIGIN since ORIGIN[0] == 0
    ORIGIN = (0, 1)


class BuckelsDither(Dither):
    # 0 * 2 1
    # 1 2 1 0
    # 0 1 0 0
    PATTERN = np.array(((0, 0, 2, 1), (1, 2, 1, 0), (0, 1, 0, 0)),
                       dtype=np.float32).reshape(3, 4, 1) / np.float32(8)
    ORIGIN = (0, 1)


class JarvisDither(Dither):
    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array(((0, 0, 0, 7, 5), (3, 5, 7, 5, 3), (1, 3, 5, 3, 1)),
                       dtype=np.float32).reshape(3, 5, 1) / np.float32(48)
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
        SRGBResize(im, (screen.X_RES, screen.Y_RES), Image.LANCZOS))


@functools.lru_cache(None)
def lookahead_options(screen, lookahead, last_pixel_4bit, x):
    options_4bit = np.empty((2 ** lookahead, lookahead), dtype=np.uint8)
    options_rgb = np.empty((2 ** lookahead, lookahead, 3), dtype=np.float32)
    for i in range(2 ** lookahead):
        output_pixel_4bit = last_pixel_4bit
        for j in range(lookahead):
            xx = x + j
            palette_choices_4bit, palette_choices_rgb = \
                screen.pixel_palette_options(output_pixel_4bit, xx)
            output_pixel_4bit = palette_choices_4bit[(i & (1 << j)) >> j]
            output_pixel_rgb = np.array(
                palette_choices_rgb[(i & (1 << j)) >> j])
            # XXX copy
            options_4bit[i, j] = output_pixel_4bit
            options_rgb[i, j, :] = np.copy(output_pixel_rgb)

    return options_4bit, options_rgb


def dither_lookahead(
        screen: Screen, image_rgb: np.ndarray, dither: Dither, differ:
        ColourDistance, x, y, last_pixel_4bit, lookahead
) -> Tuple[np.ndarray, np.ndarray]:
    el, er, xl, xr = dither.x_dither_bounds(screen, x)

    # X coord value of larger of dither bounding box or lookahead horizon
    xxr = min(max(x + lookahead, xr), screen.X_RES)

    # copies of input pixels so we can dither in bulk
    # Leave enough space so we can dither the last of our lookahead pixels
    lah_image_rgb = np.zeros(
        (2 ** lookahead, lookahead + xr - xl, 3), dtype=np.float32)
    lah_image_rgb[:, 0:xxr - x, :] = np.copy(image_rgb[y, x:xxr, :])

    options_4bit, options_rgb = lookahead_options(
        screen, lookahead, last_pixel_4bit, x % 4)
    for i in range(xxr - x):
        # options_rgb choices are fixed, but we can still distribute
        # quantization error from having made these choices, in order to compute
        # the total error
        input_pixels = np.copy(lah_image_rgb[:, i, :])
        output_pixels = options_rgb[:, i, :]
        quant_error = input_pixels - output_pixels
        # Don't update the input at position x (since we've already chosen
        # fixed outputs), but do propagate quantization errors to positions >x
        # so we can compensate for how good/bad these choices were
        # XXX vectorize
        for j in range(2 ** lookahead):
            # print(quant_error[j])
            dither.apply_one_line(screen,
                                  lah_image_rgb[j, :, :].reshape(1, -1, 3),
                                  i, 0, quant_error[j])

    error = differ.distance(np.clip(
        lah_image_rgb[:, 0:lookahead, :], 0, 255), options_4bit)
    # print(error.dtype)
    # print(lah_image_lab)
    # print("error=", error)
    # print(error.shape)
    total_error = np.sum(np.power(error, 2), axis=1)
    # print("total_error=", total_error)
    best = np.argmin(total_error)
    # print("best=", best)
    # print("best 4bit=", options_4bit[best, 0].item(), options_rgb[best, 0, :])
    return options_4bit[best, 0].item(), options_rgb[best, 0, :]


def dither_image(
        screen: Screen, image_rgb: np.ndarray, dither: Dither, differ:
        ColourDistance, lookahead) -> Tuple[np.ndarray, np.ndarray]:
    image_4bit = np.empty(
        (image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

    # pattern = dither.PATTERN

    for y in range(screen.Y_RES):
        print(y)
        output_pixel_4bit = np.uint8(0)
        for x in range(screen.X_RES):
        # for x in range(pattern.ORIGIN[1], pattern.ORIGIN[1] + screen.X_RES):
            input_pixel_rgb = np.copy(image_rgb[y, x, :])
            output_pixel_4bit, output_pixel_rgb = \
                dither_lookahead(screen, image_rgb, dither, differ, x, y,
                                 output_pixel_4bit, lookahead)
            image_4bit[y, x] = output_pixel_4bit
            image_rgb[y, x, :] = output_pixel_rgb
            quant_error = input_pixel_rgb - output_pixel_rgb
            dither.apply(screen, image_rgb, x, y, quant_error)

    return image_4bit, image_rgb


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
    # image_rgb.show()

    # dither = FloydSteinbergDither()
    # dither = BuckelsDither()
    dither = JarvisDither()

    differ = CIE2000Distance()

    output_4bit, output_rgb = dither_image(screen, image, dither, differ,
                                           lookahead=args.lookahead)
    screen.pack(output_4bit)

    out_image = Image.fromarray(linear_to_srgb(output_rgb).astype(np.uint8))
    outfile = os.path.join(os.path.splitext(args.output)[0] + ".png")
    out_image.save(outfile, "PNG")
    out_image.show(title=outfile)
    # bitmap = Image.fromarray(screen.bitmap.astype('uint8') * 255)

    with open(args.output, "wb") as f:
        f.write(bytes(screen.main))
        f.write(bytes(screen.aux))


if __name__ == "__main__":
    main()
