import argparse
from PIL import Image
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath import color_diff

import numpy as np

X_RES = 560
Y_RES = 192

#         for each y from top to bottom
#            for each x from left to right
#               oldpixel  := pixel[x][y]
#               newpixel  := find_closest_palette_color(oldpixel)
#               pixel[x][y]  := newpixel
#               quant_error  := oldpixel - newpixel
#               pixel[x+1][y  ] := pixel[x+1][y  ] + quant_error * 7/16
#               pixel[x-1][y+1] := pixel[x-1][y+1] + quant_error * 3/16
#               pixel[x  ][y+1] := pixel[x  ][y+1] + quant_error * 5/16
#               pixel[x+1][y+1] := pixel[x+1][y+1] + quant_error * 1/16

RGB = {
    (False, False, False, False): np.array((0, 0, 0)),  # Black
    (False, False, False, True): np.array((148, 12, 125)),  # Magenta
    (False, False, True, False): np.array((99, 77, 0)),  # Brown
    (False, False, True, True): np.array((249, 86, 29)),  # Orange
    (False, True, False, False): np.array((51, 111, 0)),  # Dark green
    (False, True, False, True): np.array((126, 126, 126)),  # Grey1
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


def find_closest_color(pixel):
    least_diff = 1e9
    best_colour = None
    for v in RGB.values():
        diff = np.sum(np.power(v - np.array(pixel), 2))
        if diff < least_diff:
            least_diff = diff
            best_colour = v
    return best_colour


def dither(filename):
    im = Image.open(filename)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.resize((X_RES, Y_RES), resample=Image.LANCZOS)
    im.show()
    for y in range(Y_RES):
        print(y)
        for x in range(X_RES):
            # print(x)
            oldpixel = im.getpixel((x, y))
            newpixel = find_closest_color(oldpixel, newpixel, x)

            quant_error = oldpixel - newpixel
            # print(quant_error)
            if x < (X_RES-1):
                im.putpixel((x + 1, y), tuple((
                    np.array(im.getpixel((x + 1, y))) + quant_error * 7 /
                    16).astype(np.int)))
            if x > 0 and y < Y_RES-1:
                im.putpixel((x - 1, y + 1),
                            tuple((np.array(im.getpixel(
                                (x - 1, y + 1))) + quant_error * 3 /
                                   16).astype(np.int)))
            if y < Y_RES-1:
                im.putpixel((x, y + 1),
                            tuple((np.array(im.getpixel(
                                (x, y + 1)) + quant_error * 5 / 16)).astype(
                                np.int)))
            if x < (X_RES-1) and y < (Y_RES-1):
                im.putpixel((x + 1, y + 1),
                            tuple((np.array(im.getpixel(
                                (x + 1, y + 1)) + quant_error * 1 /
                                            16)).astype(np.int)))
    im.show()
            #
            # c = {}
            # for value in True, False:
            #     if value:
            #         s.set(x, y)
            #     else:
            #         s.unset(x, y)
            #
            #     c[value] = convert_color(
            #         sRGBColor(*s.colours(x, y)[0], is_upscaled=True),
            #         LabColor)
            #
            # diffs = [
            #     (
            #         color_diff.delta_e_cie2000(oldpixel, newpixel),
            #         newpixel,
            #         value
            #     )
            #     for value, newpixel in c.items()]
            #
            # print(diffs)
            # diff, newpixel, value = min(diffs)
            # if value:
            #     s.set(x, y)
            # else:
            #     s.unset(x, y)
            #
            # put(im, (x, y), np.array(newpixel.get_value_tuple()))
            # yield x, y, value
            # print(oldpixel, newpixel)
            # quant_error = np.array(oldpixel.get_value_tuple()) - np.array(
            #     newpixel.get_value_tuple())
            # print("qe = %s" % quant_error)
            #
            # if x < (screen.X_RES - 1):
            #     nr = (np.array(im.getpixel((x + 1, y)), dtype=np.float) / 256 +
            #         quant_error * 7 / 16)
            #     print(nr * 256)
            #     put(im, (x + 1, y), nr)
            #     print(im.getpixel((x+1, y)))
            #     if y < (screen.Y_RES - 1):
            #         put(im, (x - 1, y + 1),
            #             np.array(im.getpixel((x - 1, y + 1)), dtype=np.float) / 256 +
            #             quant_error * 3 / 16)
            #         put(im, (x, y + 1),
            #             np.array(im.getpixel((x, y + 1)), dtype=np.float) / 256 +
            #             quant_error * 5 / 16)
            #         put(im, (x + 1, y + 1),
            #             np.array(im.getpixel((x + 1, y + 1)), dtype=np.float) / 256 +
            #             quant_error * 1 / 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file to process")

    args = parser.parse_args()
    dither(args.input)


#
# def put(image, xy, lab_value):
#     # print(lab_value)
#     image.putpixel(xy, tuple((lab_value * 256).astype(int)))

if __name__ == "__main__":
    main()
