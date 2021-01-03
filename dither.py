import argparse
import functools
from typing import Tuple

from PIL import Image
import colormath.color_conversions
import colormath.color_diff
import colormath.color_objects
import numpy as np

# TODO:
# - compare to bmp2dhr and a2bestpix
# - deal with fringing
# - look ahead N pixels and compute all 2^N bit patterns, then minimize
#   average error
# - optimize Dither.apply() critical path

X_RES = 560
Y_RES = 192


def srgb_to_linear_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** gamma)


def linear_to_srgb_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1.0 / gamma) -
                    0.055)

def srgb_to_linear(im: Image) -> Image:
    a = np.array(im, dtype=np.float32) / 255.0
    rgb_linear = srgb_to_linear_array(a, gamma=2.4)
    return Image.fromarray(
        (np.clip(rgb_linear, 0.0, 1.0) * 255).astype(np.uint8))


def linear_to_srgb(im: Image) -> Image:
    a = np.array(im, dtype=np.float32) / 255.0
    srgb = linear_to_srgb_array(a, gamma = 2.4)
    return Image.fromarray((np.clip(srgb, 0.0, 1.0) * 255).astype(np.uint8))


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

# RGB = {
#     (False, False, False, False): np.array((0, 0, 0)),  # Black
#     (False, False, False, True): np.array((189, 0, 102)),  # Magenta
#     (False, False, True, False): np.array((81, 86, 0)),  # Brown
#     (False, False, True, True): np.array((238, 55, 0)),  # Orange
#     (False, True, False, False): np.array((3, 135, 0)),  # Dark green
#     # XXX RGB values are used as keys in DOTS dict, need to be unique
#     (False, True, False, True): np.array((111, 111, 111)),  # Grey1
#     (False, True, True, False): np.array((14, 237, 0)),  # Green
#     (False, True, True, True): np.array((204, 213, 0)),  # Yellow
#     (True, False, False, False): np.array((13, 0, 242)),  # Dark blue
#     (True, False, False, True): np.array((221, 0, 241)),  # Violet
#     (True, False, True, False): np.array((112, 112, 112)),  # Grey2
#     (True, False, True, True): np.array((236, 72, 229)),  # Pink
#     (True, True, False, False): np.array((0, 157, 241)),  # Med blue
#     (True, True, False, True): np.array((142, 133, 240)),  # Light blue
#     (True, True, True, False): np.array((39, 247, 117)),  # Aqua
#     (True, True, True, True): np.array((236, 236, 236)),  # White
# }
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
#
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
    def distance(rgb1: Tuple[int], rgb2: Tuple[int]) -> float:
        raise NotImplementedError


class RGBDistance(ColourDistance):
    """Euclidean squared distance in RGB colour space."""

    @staticmethod
    def distance(rgb1: Tuple[int], rgb2: Tuple[int]) -> float:
        return float(np.asscalar(np.sum(np.power(np.array(rgb1) - np.array(
            rgb2), 2))))


class CIE2000Distance(ColourDistance):
    """CIE2000 delta-E distance."""

    @staticmethod
    @functools.lru_cache(None)
    def _to_lab(rgb):
        srgb = np.clip(linear_to_srgb_array(np.array(rgb) / 255), 0.0,
                       1.0) * 255
        srgb = colormath.color_objects.sRGBColor(*tuple(srgb), is_upscaled=True)
        lab = colormath.color_conversions.convert_color(
            srgb, colormath.color_objects.LabColor)
        return lab

    def distance(self, rgb1: Tuple[int], rgb2: Tuple[int]) -> float:
        lab1 = self._to_lab(rgb1)
        lab2 = self._to_lab(rgb2)
        return colormath.color_diff.delta_e_cie2000(lab1, lab2)


class CCIR601Distance(ColourDistance):
    @staticmethod
    def _to_luma(rgb):
        return rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114

    def distance(self, rgb1: Tuple[int], rgb2: Tuple[int]) -> float:
        delta_rgb = ((rgb1[0] - rgb2[0])/255, (rgb1[1] - rgb2[1])/255,
                     (rgb1[2] - rgb2[2])/255)
        luma_diff = (self._to_luma(rgb1) - self._to_luma(rgb2)) / 255

        return (
                       delta_rgb[0] * delta_rgb[0] * 0.299 +
                       delta_rgb[1] * delta_rgb[1] * 0.587 +
                       delta_rgb[2] * delta_rgb[2] * 0.114) * 0.75 + (
                       luma_diff * luma_diff)


def find_closest_color(pixel, last_pixel, x: int):
    least_diff = 1e9
    best_colour = None

    last_dots = DOTS[tuple(last_pixel)]
    other_dots = list(last_dots)
    other_dots[x % 4] = not other_dots[x % 4]
    other_dots = tuple(other_dots)
    for v in (RGB[last_dots], RGB[other_dots]):
        diff = np.sum(np.power(v - np.array(pixel), 2))
        if diff < least_diff:
            least_diff = diff
            best_colour = v
    return best_colour


def find_closest_color(pixel, last_pixel, x: int, differ: ColourDistance):
    least_diff = 1e9
    best_colour = None

    for v in RGB.values():
        diff = differ.distance(tuple(v), pixel)
        if diff < least_diff:
            least_diff = diff
            best_colour = v
    return best_colour


class Dither:
    PATTERN = None
    ORIGIN = None

    def apply(self, image, x, y, quant_error):
        for offset, error_fraction in np.ndenumerate(self.PATTERN / np.sum(
                self.PATTERN)):
            xx = x + offset[1] - self.ORIGIN[1]
            yy = y + offset[0] - self.ORIGIN[0]
            if xx < 0 or yy < 0 or xx > (X_RES // 4 - 1) or yy > (Y_RES - 1):
                continue
            new_pixel = image.getpixel((xx, yy)) + error_fraction * quant_error
            image.putpixel((xx, yy), tuple(new_pixel.astype(int)))


class FloydSteinbergDither(Dither):
    # 0 * 7
    # 3 5 1
    PATTERN = np.array(((0, 0, 7), (3, 5, 1)))
    ORIGIN = (0, 1)


class BuckelsDither(Dither):
    # 0 * 2 1
    # 1 2 1 0
    # 0 1 0 0
    PATTERN = np.array(((0, 0, 2, 1), (1, 2, 1, 0), (0, 1, 0, 0)))
    ORIGIN = (0, 1)


class JarvisDither(Dither):
    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array(((0, 0, 0, 7, 5), (3, 5, 7, 5, 3), (1, 3, 5, 3, 1)))
    ORIGIN = (0, 2)


# XXX needed?
def SRGBResize(im, size, filter):
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
    # Convert to 8-bit
    arrOut = np.uint8(np.rint(arrOut * 255.0))
    # Convert back to PIL
    return Image.fromarray(arrOut)


def open_image(filename: str) -> Image:
    im = Image.open(filename)
    if im.mode != "RGB":
        im = im.convert("RGB")
    # rgb_linear = srgb_to_linear(np.array(im, dtype=np.float32) / 255.0)
    # im = Image.fromarray(rgb_linear * 255)
    return srgb_to_linear(SRGBResize(im, (X_RES // 4, Y_RES), Image.LANCZOS))
    # return SRGBResize(im, (X_RES // 4, Y_RES), Image.LANCZOS)


def dither_image(image: Image, dither: Dither, differ: ColourDistance) -> Image:
    for y in range(Y_RES):
        print(y)
        new_pixel = (0, 0, 0)
        for x in range(X_RES // 4):
            old_pixel = image.getpixel((x, y))
            new_pixel = find_closest_color(old_pixel, new_pixel, x, differ)
            image.putpixel((x, y), tuple(new_pixel))
            quant_error = old_pixel - new_pixel
            dither.apply(image, x, y, quant_error)
    return image


class Screen:
    def __init__(self, image: Image):
        self.bitmap = np.zeros((Y_RES, X_RES), dtype=np.bool)

        self.main = np.zeros(8192, dtype=np.uint8)
        self.aux = np.zeros(8192, dtype=np.uint8)

        for y in range(Y_RES):
            for x in range(X_RES // 4):
                pixel = image.getpixel((x, y))
                dots = DOTS[pixel]
                # phase = x % 4
                # self.bitmap[y, x] = dots[phase]
                self.bitmap[y, x * 4:(x + 1) * 4] = dots

    @staticmethod
    def y_to_base_addr(y: int) -> int:
        """Maps y coordinate to screen memory base address."""
        a = y // 64
        d = y - 64 * a
        b = d // 8
        c = d - 8 * b

        return 1024 * c + 128 * b + 40 * a

    def pack(self):
        # The DHGR display encodes 7 pixels across interleaved 4-byte sequences
        # of AUX and MAIN memory, as follows:
        # PBBBAAAA PDDCCCCB PFEEEEDD PGGGGFFF
        # Aux N    Main N   Aux N+1  Main N+1  (N even)
        main_col = np.zeros((Y_RES, X_RES // 14), dtype=np.uint8)
        aux_col = np.zeros((Y_RES, X_RES // 14), dtype=np.uint8)
        for byte_offset in range(80):
            column = np.zeros(Y_RES, dtype=np.uint8)
            for bit in range(7):
                column |= (self.bitmap[:, 7 * byte_offset + bit].astype(
                    np.uint8) << bit)
            if byte_offset % 2 == 0:
                aux_col[:, byte_offset // 2] = column
            else:
                main_col[:, (byte_offset - 1) // 2] = column

        for y in range(Y_RES):
            addr = self.y_to_base_addr(y)
            self.aux[addr:addr + 40] = aux_col[y, :]
            self.main[addr:addr + 40] = main_col[y, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file to process")
    parser.add_argument("output", type=str, help="Output file for ")

    args = parser.parse_args()
    image = open_image(args.input)

    image.show()

    # dither = FloydSteinbergDither()
    # dither = BuckelsDither()
    dither = JarvisDither()

    # differ = CIE2000Distance()
    differ = CCIR601Distance()

    output = dither_image(image, dither, differ)
    # output.show()
    screen = Screen(output)
    linear_to_srgb(output).show()
    # bitmap = Image.fromarray(screen.bitmap.astype('uint8') * 255)
    screen.pack()

    with open(args.output, "wb") as f:
        f.write(screen.main)
        f.write(screen.aux)


if __name__ == "__main__":
    main()
