import argparse
import functools
from typing import Tuple

from PIL import Image
import colormath.color_conversions
import colormath.color_diff
import colormath.color_objects
import numpy as np


# TODO:
# - switch to colours library
# - only lookahead for 560px
# - vectorize colour differences
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

DOTS = {}
for k, v in RGB.items():
    DOTS[tuple(v)] = k


class ColourDistance:
    @staticmethod
    def distance(self, rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        raise NotImplementedError


class RGBDistance(ColourDistance):
    """Euclidean squared distance in RGB colour space."""

    @staticmethod
    def distance(self, rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        return float(np.asscalar(np.sum(np.power(np.array(rgb1) - np.array(
            rgb2), 2))))


class CIE2000Distance(ColourDistance):
    """CIE2000 delta-E distance."""

    @staticmethod
    def _to_lab(rgb: Tuple[float]):
        srgb = np.clip(
            linear_to_srgb_array(np.array(rgb, dtype=np.float32) / 255), 0.0,
            1.0)
        srgb_color = colormath.color_objects.sRGBColor(*tuple(srgb),
                                                       is_upscaled=False)
        lab = colormath.color_conversions.convert_color(
            srgb_color, colormath.color_objects.LabColor)
        return lab

    def distance(self, rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        lab1 = self._to_lab(tuple(rgb1))
        lab2 = self._to_lab(tuple(rgb2))
        return colormath.color_diff.delta_e_cie2000(lab1, lab2)


class CCIR601Distance(ColourDistance):
    @staticmethod
    def _to_luma(rgb: np.ndarray):
        return rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114

    def distance(self, rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        delta_rgb = ((rgb1[0] - rgb2[0]) / 255, (rgb1[1] - rgb2[1]) / 255,
                     (rgb1[2] - rgb2[2]) / 255)
        luma_diff = (self._to_luma(rgb1) - self._to_luma(rgb2)) / 255

        # TODO: this is the formula bmp2dhr uses but what motivates it?
        return (
                       delta_rgb[0] * delta_rgb[0] * 0.299 +
                       delta_rgb[1] * delta_rgb[1] * 0.587 +
                       delta_rgb[2] * delta_rgb[2] * 0.114) * 0.75 + (
                       luma_diff * luma_diff)


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
    def find_closest_color(pixel, palette_options, differ: ColourDistance):
        least_diff = 1e9
        best_colour = None

        for v in palette_options:
            diff = differ.distance(tuple(v), pixel)
            if diff < least_diff:
                least_diff = diff
                best_colour = v
        return best_colour


class DHGR140Screen(Screen):
    """DHGR screen ignoring colour fringing, i.e. treating as 140x192x16."""

    X_RES = 140
    Y_RES = 192
    X_PIXEL_WIDTH = 4

    def _image_to_bitmap(self, image: np.ndarray) -> np.ndarray:
        bitmap = np.zeros(
            (self.Y_RES, self.X_RES * self.X_PIXEL_WIDTH), dtype=np.bool)
        for y in range(self.Y_RES):
            for x in range(self.X_RES):
                pixel = image[y, x]
                dots = DOTS[pixel]
                bitmap[y, x * self.X_PIXEL_WIDTH:(
                        (x + 1) * self.X_PIXEL_WIDTH)] = dots
        return bitmap

    @staticmethod
    def pixel_palette_options(last_pixel, x: int):
        return RGB.values()


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

    def pixel_palette_options(self, last_pixel, x: int):
        last_dots = DOTS[tuple(last_pixel)]
        other_dots = list(last_dots)
        other_dots[x % 4] = not other_dots[x % 4]
        other_dots = tuple(other_dots)
        return RGB[last_dots], RGB[other_dots]


class Dither:
    PATTERN = None
    ORIGIN = None

    def apply(self, screen: Screen, image: np.ndarray, x: int, y: int,
              quant_error: np.ndarray):
        pshape = self.PATTERN.shape
        error = self.PATTERN.reshape(
            (pshape[0], pshape[1], 1)) * quant_error.reshape((1, 1,
                                                              3))
        # print(quant_error)
        et = max(self.ORIGIN[0] - y, 0)
        eb = min(pshape[0], screen.Y_RES - 1 - y)
        el = max(self.ORIGIN[1] - x, 0)
        er = min(pshape[1], screen.X_RES - 1 - x)
        # print(x, et, eb, el, er)

        yt = y - self.ORIGIN[0] + et
        yb = y - self.ORIGIN[0] + eb
        xl = x - self.ORIGIN[1] + el
        xr = x - self.ORIGIN[1] + er
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


# XXX
def dither_one_pixel(screen: Screen, differ: ColourDistance,
                     input_pixel, last_pixel, x) -> Tuple[int]:
    palette_choices = screen.pixel_palette_options(last_pixel, x)
    return screen.find_closest_color(input_pixel, palette_choices,
                                     differ)


def dither_lookahead(
        screen: Screen, image: np.ndarray, dither: Dither, differ:
        ColourDistance,
        x, y, last_pixel, lookahead
) -> Image:
    best_error = 1e9
    best_pixel = None
    for i in range(2 ** lookahead):
        temp_image = np.empty_like(image)
        # XXX
        temp_image[y:y + 3, :, :] = image[y:y + 3, :, :]
        output_pixel = last_pixel
        total_error = 0.0
        choices = []
        inputs = []
        for j in range(min(lookahead, screen.X_RES - x)):
            xx = x + j
            input_pixel = temp_image[y, xx, :]
            palette_choices = screen.pixel_palette_options(output_pixel, xx)
            output_pixel = np.array(palette_choices[(i & (1 << j)) >> j])
            inputs.append(input_pixel)
            choices.append(output_pixel)
            # output_pixel = dither_one_pixel(screen, differ,
            #                                 input_pixel, output_pixel, xx)
            quant_error = input_pixel - output_pixel
            # TODO: try squared error
            total_error += differ.distance(input_pixel, output_pixel)
            dither.apply(screen, temp_image, xx, y, quant_error)
        # print(bin(i), total_error, inputs, choices)
        if total_error < best_error:
            best_error = total_error
            best_pixel = choices[0]
    # print(best_error, best_pixel)
    return best_pixel


def dither_image(
        screen: Screen, image: np.ndarray, dither: Dither, differ:
        ColourDistance, lookahead) -> np.ndarray:
    for y in range(screen.Y_RES):
        print(y)
        output_pixel = (0, 0, 0)
        for x in range(screen.X_RES):
            # print(x)
            input_pixel = image[y, x, :]
            output_pixel = dither_lookahead(screen, image, dither, differ, x,
                                            y, output_pixel, lookahead)
            # output_pixel = dither_one_pixel(screen, differ, input_pixel,
            #                                output_pixel, x)
            quant_error = input_pixel - output_pixel
            image[y, x, :] = output_pixel
            dither.apply(screen, image, x, y, quant_error)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file to process")
    parser.add_argument("output", type=str, help="Output file for ")
    # screen = DHGR140Screen()
    screen = DHGR560Screen()

    args = parser.parse_args()
    image = open_image(screen, args.input)
    # image.show()

    # dither = FloydSteinbergDither()
    # dither = BuckelsDither()
    dither = JarvisDither()

    differ = CIE2000Distance()
    # differ = CCIR601Distance()

    output = dither_image(screen, image, dither, differ, lookahead=1)
    screen.pack(output)

    out_image = Image.fromarray(linear_to_srgb(output).astype(np.uint8))
    out_image.show()
    # bitmap = Image.fromarray(screen.bitmap.astype('uint8') * 255)

    with open(args.output, "wb") as f:
        f.write(bytes(screen.main))
        f.write(bytes(screen.aux))


if __name__ == "__main__":
    main()
