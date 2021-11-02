"""Precomputes all possible colours available via NTSC emulation."""

import numpy as np
from PIL import Image
import screen


def main():
    s = screen.DHGRScreen(palette=None)

    colours = {}
    unique = set()

    print("import numpy as np")
    print()
    print("# Indexed by (trailing 8-bit dot pattern, x % 4)")
    print("SRGB = {")
    # For each sequence of 8 pixels, compute the RGB colour of the right-most
    # pixel, using NTSC emulation.
    # Double Hi-Res has a timing shift that rotates the displayed bits one
    # position with respect to NTSC phase.
    # TODO: should be 3?  Do I have a compensating off-by-one in bitmap_to_ntsc?
    ntsc_shift = 2
    for j in range(ntsc_shift, ntsc_shift + 4):
        bitmap = np.zeros((1, 11 + ntsc_shift), dtype=bool)
        for bits in range(256):
            bits8 = np.empty((8,), dtype=bool)
            for i in range(8):
                bits8[i] = bits & (1 << i)

            bitmap[0, j:j + 8] = bits8

            # bitmap_to_ntsc produces 3 output pixels for each DHGR input
            ntsc = s.bitmap_to_image_ntsc(bitmap)
            last_colour = ntsc[0, 3 * (j + 8) - 1, :]
            colours[(bits, j - ntsc_shift)] = last_colour
            unique.add(tuple(last_colour))
            print("  (%d, %d): np.array((%d, %d, %d))," % (
                bits, j - ntsc_shift, last_colour[0], last_colour[1],
                last_colour[2]))
    print("}")
    print("# %d unique colours" % len(unique))

    # Show spectrum of available colours sorted by HSV hue value
    im = np.zeros((128 * 4, 256 * 16, 3), dtype=np.uint8)
    for x, j in colours:
        im[128 * j:128 * (j + 1), x * 16: (x + 1) * 16, :] = colours[x, j]

    Image.fromarray(im).show()


if __name__ == "__main__":
    main()
