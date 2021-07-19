"""RGB colour palettes to target for Apple II image conversions."""

import colour
import numpy as np
import image
import palette_ntsc


class Palette:
    RGB = {}
    SRGB = None
    CAM16UCS = {}
    XYZ = {}
    DOTS = {}
    DOTS_TO_INDEX = {}
    DISTANCES_PATH = None

    # How many successive screen pixels are used to compute output pixel 
    # palette index.
    PALETTE_DEPTH = None

    def __init__(self, load_distances=True):
        # if load_distances:
        #     # CIE2000 colour distance matrix from 24-bit RGB tuple to 4-bit
        #     # palette colour.
        #     self.distances = np.memmap(self.DISTANCES_PATH, mode="r+",
        #                                dtype=np.uint8, shape=(16777216,
        #                                                       len(self.SRGB)))

        self.RGB = {}
        for k, v in self.SRGB.items():
            self.RGB[k] = (np.clip(image.srgb_to_linear_array(v / 255), 0.0,
                                   1.0) * 255).astype(np.uint8)
            self.CAM16UCS[k] = colour.convert(
                v / 255, "sRGB", "CAM16UCS").astype(np.float32)
            # self.XYZ[k] = colour.convert(v / 255, "sRGB",
            #                                  "CIE XYZ").astype(np.float32)

        # print(self.CAM02UCS)

        # Maps palette values to screen dots.  Note that these are the same as
        # the binary index values in reverse order.
        for i in range(1 << self.PALETTE_DEPTH):
            self.DOTS[i] = tuple(
                bool(i & (1 << j)) for j in range(self.PALETTE_DEPTH))

        # Reverse mapping from screen dots to palette index.
        self.DOTS_TO_INDEX = {}
        for k, v in self.DOTS.items():
            self.DOTS_TO_INDEX[v] = k


# class ToHgrPalette(Palette):
#     """4-bit palette used as default by other DHGR image converters."""
#     DISTANCES_PATH = "data/distances_tohgr.data"
#     PALETTE_DEPTH = 4
#
#     # Default tohgr/bmp2dhr palette
#     SRGB = {
#         0: np.array((0, 0, 0)),  # Black
#         8: np.array((148, 12, 125)),  # Magenta
#         4: np.array((99, 77, 0)),  # Brown
#         12: np.array((249, 86, 29)),  # Orange
#         2: np.array((51, 111, 0)),  # Dark green
#         10: np.array((126, 126, 126)),  # Grey2
#         6: np.array((67, 200, 0)),  # Green
#         14: np.array((221, 206, 23)),  # Yellow
#         1: np.array((32, 54, 212)),  # Dark blue
#         9: np.array((188, 55, 255)),  # Violet
#         5: np.array((126, 126, 126)),  # Grey1
#         13: np.array((255, 129, 236)),  # Pink
#         3: np.array((7, 168, 225)),  # Med blue
#         11: np.array((158, 172, 255)),  # Light blue
#         7: np.array((93, 248, 133)),  # Aqua
#         15: np.array((255, 255, 255)),  # White
#     }
#
#
# class OpenEmulatorPalette(Palette):
#     """4-bit palette chosen to approximately match OpenEmulator output."""
#     DISTANCES_PATH = "data/distances_openemulator.data"
#     PALETTE_DEPTH = 4
#
#     # OpenEmulator
#     SRGB = {
#         0: np.array((0, 0, 0)),  # Black
#         8: np.array((203, 0, 121)),  # Magenta
#         4: np.array((99, 103, 0)),  # Brown
#         12: np.array((244, 78, 0)),  # Orange
#         2: np.array((0, 150, 0)),  # Dark green
#         10: np.array((130, 130, 130)),  # Grey2
#         6: np.array((0, 235, 0)),  # Green
#         14: np.array((214, 218, 0)),  # Yellow
#         1: np.array((20, 0, 246)),  # Dark blue
#         9: np.array((230, 0, 244)),  # Violet
#         5: np.array((130, 130, 130)),  # Grey1
#         13: np.array((244, 105, 235)),  # Pink
#         3: np.array((0, 174, 243)),  # Med blue
#         11: np.array((160, 156, 244)),  # Light blue
#         7: np.array((25, 243, 136)),  # Aqua
#         15: np.array((244, 247, 244)),  # White
#     }
#
#
# class VirtualIIPalette(Palette):
#     """4-bit palette exactly matching Virtual II emulator output."""
#     DISTANCES_PATH = "data/distances_virtualii.data"
#     PALETTE_DEPTH = 4
#
#     SRGB = {
#         0: np.array((0, 0, 0)),  # Black
#         8: np.array((231, 36, 66)),  # Magenta
#         4: np.array((154, 104, 0)),  # Brown
#         12: np.array((255, 124, 0)),  # Orange
#         2: np.array((0, 135, 45)),  # Dark green
#         10: np.array((104, 104, 104)),  # Grey2
#         6: np.array((0, 222, 0)),  # Green
#         14: np.array((255, 252, 0)),  # Yellow
#         1: np.array((1, 30, 169)),  # Dark blue
#         9: np.array((230, 73, 228)),  # Violet
#         5: np.array((185, 185, 185)),  # Grey1
#         13: np.array((255, 171, 153)),  # Pink
#         3: np.array((47, 69, 255)),  # Med blue
#         11: np.array((120, 187, 255)),  # Light blue
#         7: np.array((83, 250, 208)),  # Aqua
#         15: np.array((255, 255, 255)),  # White
#     }


class NTSCPalette(Palette):
    """8-bit NTSC palette computed by averaging chroma signal over 8 pixels."""
    DISTANCES_PATH = 'data/distances_ntsc.data'
    PALETTE_DEPTH = 8

    # Computed using ntsc_colours.py
    SRGB = palette_ntsc.SRGB


PALETTES = {
    # 'openemulator': OpenEmulatorPalette,
    # 'virtualii': VirtualIIPalette,
    # 'tohgr': ToHgrPalette,
    'ntsc': NTSCPalette
}

DEFAULT_PALETTE = 'ntsc'
