import argparse
from PIL import Image

import numpy as np

# TODO:
# - output binary files that can be viewed on Apple II
# - use perceptual colour difference model
# - look ahead N pixels and compute all 2^N bit patterns, then minimize
#   average error
# - optimize Dither.apply() critical path

X_RES = 560
Y_RES = 192

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

NAMES = {
    (0, 0, 0): "Black",
    (148, 12, 125): "Magenta",
    (99, 77, 0): "Brown",
    (249, 86, 29): "Orange",
    (51, 111, 0): "Dark green",
    (126, 126, 125): "Grey1",  # XXX
    (67, 200, 0): "Green",
    (221, 206, 23): "Yellow",
    (32, 54, 212): "Dark blue",
    (188, 55, 255): "Violet",
    (126, 126, 126): "Grey2",
    (255, 129, 236): "Pink",
    (7, 168, 225): "Med blue",
    (158, 172, 255): "Light blue",
    (93, 248, 133): "Aqua",
    (255, 255, 255): "White"
}

DOTS = {}
for k, v in RGB.items():
    DOTS[tuple(v)] = k


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


class Dither:
    PATTERN = None
    ORIGIN = None

    def apply(self, image, x, y, quant_error):
        for offset, error_fraction in np.ndenumerate(self.PATTERN / np.sum(
                self.PATTERN)):
            xx = x + offset[1] - self.ORIGIN[1]
            yy = y + offset[0] - self.ORIGIN[0]
            if xx < 0 or yy < 0 or xx > (X_RES - 1) or yy > (Y_RES - 1):
                continue
            new_pixel = image.getpixel((xx, yy)) + error_fraction * quant_error
            image.putpixel((xx, yy), tuple(new_pixel.astype(int)))


class FloydSteinbergDither(Dither):
    # 0 * 7
    # 3 5 1
    PATTERN = np.array(((0, 0, 7), (3, 5, 1)))
    ORIGIN = (0, 1)


class KennawayDither(Dither):
    # 0 * 7 5 3 1
    # 3 5 3 1 1 0
    PATTERN = np.array(((0, 0, 7, 5, 3, 1), (3, 5, 3, 1, 1, 0)))
    ORIGIN = (0, 1)


def dither(filename):
    im = Image.open(filename)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.resize((X_RES, Y_RES), resample=Image.LANCZOS)
    im.show()

    # ditherer = FloydSteinbergDither()
    ditherer = KennawayDither()
    for y in range(Y_RES):
        print(y)
        newpixel = (0, 0, 0)
        for x in range(X_RES):
            oldpixel = im.getpixel((x, y))
            newpixel = find_closest_color(oldpixel, newpixel, x)
            im.putpixel((x, y), tuple(newpixel))
            quant_error = oldpixel - newpixel
            ditherer.apply(im, x, y, quant_error)
    im.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file to process")

    args = parser.parse_args()
    dither(args.input)


if __name__ == "__main__":
    main()
