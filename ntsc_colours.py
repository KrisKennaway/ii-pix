"""Precomputes all possible colours available via NTSC emulation."""

import colour
import numpy as np
from PIL import Image
import screen


def main():
    s = screen.DHGR560Screen(palette=None)
    bitmap = np.zeros((1, 8), dtype=np.bool)

    colours = {}
    unique = set()

    print("SRGB = {")
    # For each sequence of 8 pixels, compute the RGB colour of the right-most
    # pixel, using NTSC emulation.
    for bits in range(256):
        bits8 = np.empty((8,), dtype=np.bool)
        for i in range(8):
            bits8[i] = bits & (1 << i)
        # Double Hi-Res has a timing shift that rotates the displayed bits one
        # position with respect to NTSC phase.
        bitmap[0, :] = bits8[[3, 0, 1, 2, 7, 4, 5, 6]]

        ntsc = s.bitmap_to_ntsc(bitmap)
        last_colour = ntsc[0, -1, :]
        colours[bits] = last_colour
        unique.add(tuple(last_colour))
        print("%d: np.array((%d, %d, %d))," % (
            bits, last_colour[0], last_colour[1], last_colour[2]))
    print("}")
    print("# %d unique colours" % len(unique))

    # Show spectrum of available colours sorted by HSV hue value
    im = np.zeros((128, 256 * 16, 3), dtype=np.uint8)
    for x, hsv in enumerate(sorted([tuple(colour.RGB_to_HSV(c / 256)) for c in
                                    colours.values()])):
        im[0:128, x * 16: (x + 1) * 16, :] = colour.HSV_to_RGB(hsv) * 256

    Image.fromarray(im).show()


if __name__ == "__main__":
    main()
