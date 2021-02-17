"""RGB colour palettes to target for Apple II image conversions."""

import numpy as np
import image


class Palette:
    RGB = {}
    SRGB = None
    DOTS = {}
    DOTS_TO_INDEX = {}
    DISTANCES_PATH = None

    # How many successive screen pixels are used to compute output pixel 
    # palette index.
    PALETTE_DEPTH = None

    def __init__(self, load_distances=True):
        if load_distances:
            # CIE2000 colour distance matrix from 24-bit RGB tuple to 4-bit
            # palette colour.
            self.distances = np.memmap(self.DISTANCES_PATH, mode="r+",
                                       dtype=np.uint8, shape=(16777216,
                                                              len(self.SRGB)))

        self.RGB = {}
        for k, v in self.SRGB.items():
            self.RGB[k] = (np.clip(image.srgb_to_linear_array(v / 255), 0.0,
                                   1.0) * 255).astype(np.uint8)

        # Maps palette values to screen dots.  Note that these are the same as
        # the binary index values in reverse order.
        for i in range(1 << self.PALETTE_DEPTH):
            self.DOTS[i] = tuple(
                bool(i & (1 << j)) for j in range(self.PALETTE_DEPTH))

        # Reverse mapping from screen dots to palette index.
        self.DOTS_TO_INDEX = {}
        for k, v in self.DOTS.items():
            self.DOTS_TO_INDEX[v] = k


class ToHgrPalette(Palette):
    DISTANCES_PATH = "data/distances_tohgr.data"
    PALETTE_DEPTH = 4

    # Default tohgr/bmp2dhr palette
    SRGB = {
        0: np.array((0, 0, 0)),  # Black
        8: np.array((148, 12, 125)),  # Magenta
        4: np.array((99, 77, 0)),  # Brown
        12: np.array((249, 86, 29)),  # Orange
        2: np.array((51, 111, 0)),  # Dark green
        10: np.array((126, 126, 126)),  # Grey2
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


class OpenEmulatorPalette(Palette):
    DISTANCES_PATH = "data/distances_openemulator.data"
    PALETTE_DEPTH = 4

    # OpenEmulator
    SRGB = {
        0: np.array((0, 0, 0)),  # Black
        8: np.array((203, 0, 121)),  # Magenta
        4: np.array((99, 103, 0)),  # Brown
        12: np.array((244, 78, 0)),  # Orange
        2: np.array((0, 150, 0)),  # Dark green
        10: np.array((130, 130, 130)),  # Grey2
        6: np.array((0, 235, 0)),  # Green
        14: np.array((214, 218, 0)),  # Yellow
        1: np.array((20, 0, 246)),  # Dark blue
        9: np.array((230, 0, 244)),  # Violet
        5: np.array((130, 130, 130)),  # Grey1
        13: np.array((244, 105, 235)),  # Pink
        3: np.array((0, 174, 243)),  # Med blue
        11: np.array((160, 156, 244)),  # Light blue
        7: np.array((25, 243, 136)),  # Aqua
        15: np.array((244, 247, 244)),  # White
    }


class VirtualIIPalette(Palette):
    DISTANCES_PATH = "data/distances_virtualii.data"
    PALETTE_DEPTH = 4

    SRGB = {
        0: np.array((0, 0, 0)),  # Black
        8: np.array((231, 36, 66)),  # Magenta
        4: np.array((154, 104, 0)),  # Brown
        12: np.array((255, 124, 0)),  # Orange
        2: np.array((0, 135, 45)),  # Dark green
        10: np.array((104, 104, 104)),  # Grey2
        6: np.array((0, 222, 0)),  # Green
        14: np.array((255, 252, 0)),  # Yellow
        1: np.array((1, 30, 169)),  # Dark blue
        9: np.array((230, 73, 228)),  # Violet
        5: np.array((185, 185, 185)),  # Grey1
        13: np.array((255, 171, 153)),  # Pink
        3: np.array((47, 69, 255)),  # Med blue
        11: np.array((120, 187, 255)),  # Light blue
        7: np.array((83, 250, 208)),  # Aqua
        15: np.array((255, 255, 255)),  # White
    }


class NTSCPalette(Palette):
    DISTANCES_PATH = 'data/distances_ntsc.data'
    PALETTE_DEPTH = 8

    # Computed using ntsc_colours.py
    SRGB = {
        0: np.array((0, 0, 0)),
        1: np.array((0, 0, 115)),
        2: np.array((0, 37, 0)),
        3: np.array((0, 23, 88)),
        4: np.array((14, 14, 0)),
        5: np.array((0, 0, 0)),
        6: np.array((0, 52, 0)),
        7: np.array((0, 37, 0)),
        8: np.array((64, 0, 26)),
        9: np.array((49, 0, 141)),
        10: np.array((0, 0, 0)),
        11: np.array((0, 0, 115)),
        12: np.array((79, 0, 0)),
        13: np.array((64, 0, 26)),
        14: np.array((14, 14, 0)),
        15: np.array((0, 0, 0)),
        16: np.array((49, 48, 178)),
        17: np.array((34, 34, 255)),
        18: np.array((0, 86, 152)),
        19: np.array((0, 71, 255)),
        20: np.array((63, 63, 63)),
        21: np.array((49, 48, 178)),
        22: np.array((0, 101, 37)),
        23: np.array((0, 86, 152)),
        24: np.array((113, 10, 204)),
        25: np.array((98, 0, 255)),
        26: np.array((49, 48, 178)),
        27: np.array((34, 34, 255)),
        28: np.array((128, 25, 89)),
        29: np.array((113, 10, 204)),
        30: np.array((63, 63, 63)),
        31: np.array((49, 48, 178)),
        32: np.array((0, 101, 37)),
        33: np.array((0, 86, 152)),
        34: np.array((0, 139, 11)),
        35: np.array((0, 124, 126)),
        36: np.array((13, 116, 0)),
        37: np.array((0, 101, 37)),
        38: np.array((0, 154, 0)),
        39: np.array((0, 139, 11)),
        40: np.array((63, 63, 63)),
        41: np.array((49, 48, 178)),
        42: np.array((0, 101, 37)),
        43: np.array((0, 86, 152)),
        44: np.array((78, 78, 0)),
        45: np.array((63, 63, 63)),
        46: np.array((13, 116, 0)),
        47: np.array((0, 101, 37)),
        48: np.array((48, 150, 216)),
        49: np.array((33, 135, 255)),
        50: np.array((0, 188, 190)),
        51: np.array((0, 173, 255)),
        52: np.array((62, 165, 101)),
        53: np.array((48, 150, 216)),
        54: np.array((0, 203, 75)),
        55: np.array((0, 188, 190)),
        56: np.array((112, 112, 242)),
        57: np.array((98, 97, 255)),
        58: np.array((48, 150, 216)),
        59: np.array((33, 135, 255)),
        60: np.array((127, 127, 127)),
        61: np.array((112, 112, 242)),
        62: np.array((62, 165, 101)),
        63: np.array((48, 150, 216)),
        64: np.array((78, 78, 0)),
        65: np.array((63, 63, 63)),
        66: np.array((13, 116, 0)),
        67: np.array((0, 101, 37)),
        68: np.array((93, 93, 0)),
        69: np.array((78, 78, 0)),
        70: np.array((28, 131, 0)),
        71: np.array((13, 116, 0)),
        72: np.array((142, 40, 0)),
        73: np.array((128, 25, 89)),
        74: np.array((78, 78, 0)),
        75: np.array((63, 63, 63)),
        76: np.array((157, 55, 0)),
        77: np.array((142, 40, 0)),
        78: np.array((93, 93, 0)),
        79: np.array((78, 78, 0)),
        80: np.array((127, 127, 127)),
        81: np.array((112, 112, 242)),
        82: np.array((62, 165, 101)),
        83: np.array((48, 150, 216)),
        84: np.array((142, 142, 12)),
        85: np.array((127, 127, 127)),
        86: np.array((77, 180, 0)),
        87: np.array((62, 165, 101)),
        88: np.array((192, 89, 153)),
        89: np.array((177, 74, 255)),
        90: np.array((127, 127, 127)),
        91: np.array((112, 112, 242)),
        92: np.array((206, 104, 38)),
        93: np.array((192, 89, 153)),
        94: np.array((142, 142, 12)),
        95: np.array((127, 127, 127)),
        96: np.array((77, 180, 0)),
        97: np.array((62, 165, 101)),
        98: np.array((13, 218, 0)),
        99: np.array((0, 203, 75)),
        100: np.array((92, 195, 0)),
        101: np.array((77, 180, 0)),
        102: np.array((27, 233, 0)),
        103: np.array((13, 218, 0)),
        104: np.array((142, 142, 12)),
        105: np.array((127, 127, 127)),
        106: np.array((77, 180, 0)),
        107: np.array((62, 165, 101)),
        108: np.array((156, 157, 0)),
        109: np.array((142, 142, 12)),
        110: np.array((92, 195, 0)),
        111: np.array((77, 180, 0)),
        112: np.array((126, 229, 165)),
        113: np.array((112, 214, 255)),
        114: np.array((62, 255, 138)),
        115: np.array((47, 252, 253)),
        116: np.array((141, 244, 50)),
        117: np.array((126, 229, 165)),
        118: np.array((76, 255, 23)),
        119: np.array((62, 255, 138)),
        120: np.array((191, 191, 191)),
        121: np.array((176, 176, 255)),
        122: np.array((126, 229, 165)),
        123: np.array((112, 214, 255)),
        124: np.array((205, 206, 76)),
        125: np.array((191, 191, 191)),
        126: np.array((141, 244, 50)),
        127: np.array((126, 229, 165)),
        128: np.array((128, 25, 89)),
        129: np.array((113, 10, 204)),
        130: np.array((63, 63, 63)),
        131: np.array((49, 48, 178)),
        132: np.array((142, 40, 0)),
        133: np.array((128, 25, 89)),
        134: np.array((78, 78, 0)),
        135: np.array((63, 63, 63)),
        136: np.array((192, 0, 116)),
        137: np.array((178, 0, 231)),
        138: np.array((128, 25, 89)),
        139: np.array((113, 10, 204)),
        140: np.array((207, 2, 1)),
        141: np.array((192, 0, 116)),
        142: np.array((142, 40, 0)),
        143: np.array((128, 25, 89)),
        144: np.array((177, 74, 255)),
        145: np.array((162, 59, 255)),
        146: np.array((112, 112, 242)),
        147: np.array((98, 97, 255)),
        148: np.array((192, 89, 153)),
        149: np.array((177, 74, 255)),
        150: np.array((127, 127, 127)),
        151: np.array((112, 112, 242)),
        152: np.array((241, 36, 255)),
        153: np.array((227, 21, 255)),
        154: np.array((177, 74, 255)),
        155: np.array((162, 59, 255)),
        156: np.array((255, 51, 179)),
        157: np.array((241, 36, 255)),
        158: np.array((192, 89, 153)),
        159: np.array((177, 74, 255)),
        160: np.array((127, 127, 127)),
        161: np.array((112, 112, 242)),
        162: np.array((62, 165, 101)),
        163: np.array((48, 150, 216)),
        164: np.array((142, 142, 12)),
        165: np.array((127, 127, 127)),
        166: np.array((77, 180, 0)),
        167: np.array((62, 165, 101)),
        168: np.array((192, 89, 153)),
        169: np.array((177, 74, 255)),
        170: np.array((127, 127, 127)),
        171: np.array((112, 112, 242)),
        172: np.array((206, 104, 38)),
        173: np.array((192, 89, 153)),
        174: np.array((142, 142, 12)),
        175: np.array((127, 127, 127)),
        176: np.array((176, 176, 255)),
        177: np.array((161, 161, 255)),
        178: np.array((112, 214, 255)),
        179: np.array((97, 199, 255)),
        180: np.array((191, 191, 191)),
        181: np.array((176, 176, 255)),
        182: np.array((126, 229, 165)),
        183: np.array((112, 214, 255)),
        184: np.array((241, 138, 255)),
        185: np.array((226, 123, 255)),
        186: np.array((176, 176, 255)),
        187: np.array((161, 161, 255)),
        188: np.array((255, 153, 217)),
        189: np.array((241, 138, 255)),
        190: np.array((191, 191, 191)),
        191: np.array((176, 176, 255)),
        192: np.array((206, 104, 38)),
        193: np.array((192, 89, 153)),
        194: np.array((142, 142, 12)),
        195: np.array((127, 127, 127)),
        196: np.array((221, 119, 0)),
        197: np.array((206, 104, 38)),
        198: np.array((156, 157, 0)),
        199: np.array((142, 142, 12)),
        200: np.array((255, 66, 64)),
        201: np.array((255, 51, 179)),
        202: np.array((206, 104, 38)),
        203: np.array((192, 89, 153)),
        204: np.array((255, 81, 0)),
        205: np.array((255, 66, 64)),
        206: np.array((221, 119, 0)),
        207: np.array((206, 104, 38)),
        208: np.array((255, 153, 217)),
        209: np.array((241, 138, 255)),
        210: np.array((191, 191, 191)),
        211: np.array((176, 176, 255)),
        212: np.array((255, 168, 102)),
        213: np.array((255, 153, 217)),
        214: np.array((205, 206, 76)),
        215: np.array((191, 191, 191)),
        216: np.array((255, 115, 243)),
        217: np.array((255, 100, 255)),
        218: np.array((255, 153, 217)),
        219: np.array((241, 138, 255)),
        220: np.array((255, 130, 128)),
        221: np.array((255, 115, 243)),
        222: np.array((255, 168, 102)),
        223: np.array((255, 153, 217)),
        224: np.array((205, 206, 76)),
        225: np.array((191, 191, 191)),
        226: np.array((141, 244, 50)),
        227: np.array((126, 229, 165)),
        228: np.array((220, 220, 0)),
        229: np.array((205, 206, 76)),
        230: np.array((156, 255, 0)),
        231: np.array((141, 244, 50)),
        232: np.array((255, 168, 102)),
        233: np.array((255, 153, 217)),
        234: np.array((205, 206, 76)),
        235: np.array((191, 191, 191)),
        236: np.array((255, 183, 0)),
        237: np.array((255, 168, 102)),
        238: np.array((220, 220, 0)),
        239: np.array((205, 206, 76)),
        240: np.array((254, 255, 255)),
        241: np.array((240, 240, 255)),
        242: np.array((190, 255, 228)),
        243: np.array((175, 255, 255)),
        244: np.array((255, 255, 139)),
        245: np.array((254, 255, 255)),
        246: np.array((205, 255, 113)),
        247: np.array((190, 255, 228)),
        248: np.array((255, 217, 255)),
        249: np.array((255, 202, 255)),
        250: np.array((254, 255, 255)),
        251: np.array((240, 240, 255)),
        252: np.array((255, 231, 166)),
        253: np.array((255, 217, 255)),
        254: np.array((255, 255, 139)),
        255: np.array((254, 255, 255)),
    }
    # 85 unique colours


PALETTES = {
    'openemulator': OpenEmulatorPalette,
    'virtualii': VirtualIIPalette,
    'tohgr': ToHgrPalette,
    'ntsc': NTSCPalette
}

DEFAULT_PALETTE = 'openemulator'
