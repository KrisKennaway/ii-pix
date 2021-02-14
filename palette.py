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
        1: np.array((0, 0, 62)),
        2: np.array((0, 18, 0)),
        3: np.array((0, 3, 28)),
        4: np.array((44, 14, 0)),
        5: np.array((0, 0, 0)),
        6: np.array((0, 32, 0)),
        7: np.array((0, 18, 0)),
        8: np.array((67, 0, 34)),
        9: np.array((22, 0, 96)),
        10: np.array((0, 0, 0)),
        11: np.array((0, 0, 62)),
        12: np.array((112, 0, 0)),
        13: np.array((67, 0, 34)),
        14: np.array((44, 14, 0)),
        15: np.array((0, 0, 0)),
        16: np.array((24, 54, 131)),
        17: np.array((0, 40, 193)),
        18: np.array((0, 73, 97)),
        19: np.array((0, 58, 159)),
        20: np.array((69, 69, 69)),
        21: np.array((24, 54, 131)),
        22: np.array((1, 87, 35)),
        23: np.array((0, 73, 97)),
        24: np.array((91, 36, 165)),
        25: np.array((47, 22, 227)),
        26: np.array((24, 54, 131)),
        27: np.array((0, 40, 193)),
        28: np.array((136, 50, 103)),
        29: np.array((91, 36, 165)),
        30: np.array((69, 69, 69)),
        31: np.array((24, 54, 131)),
        32: np.array((1, 87, 35)),
        33: np.array((0, 73, 97)),
        34: np.array((0, 105, 1)),
        35: np.array((0, 91, 63)),
        36: np.array((46, 101, 0)),
        37: np.array((1, 87, 35)),
        38: np.array((0, 120, 0)),
        39: np.array((0, 105, 1)),
        40: np.array((69, 69, 69)),
        41: np.array((24, 54, 131)),
        42: np.array((1, 87, 35)),
        43: np.array((0, 73, 97)),
        44: np.array((113, 83, 7)),
        45: np.array((69, 69, 69)),
        46: np.array((46, 101, 0)),
        47: np.array((1, 87, 35)),
        48: np.array((26, 142, 166)),
        49: np.array((0, 127, 228)),
        50: np.array((0, 160, 132)),
        51: np.array((0, 146, 194)),
        52: np.array((70, 156, 104)),
        53: np.array((26, 142, 166)),
        54: np.array((3, 174, 70)),
        55: np.array((0, 160, 132)),
        56: np.array((93, 124, 200)),
        57: np.array((48, 109, 255)),
        58: np.array((26, 142, 166)),
        59: np.array((0, 127, 228)),
        60: np.array((138, 138, 138)),
        61: np.array((93, 124, 200)),
        62: np.array((70, 156, 104)),
        63: np.array((26, 142, 166)),
        64: np.array((113, 83, 7)),
        65: np.array((69, 69, 69)),
        66: np.array((46, 101, 0)),
        67: np.array((1, 87, 35)),
        68: np.array((158, 97, 0)),
        69: np.array((113, 83, 7)),
        70: np.array((91, 116, 0)),
        71: np.array((46, 101, 0)),
        72: np.array((181, 65, 41)),
        73: np.array((136, 50, 103)),
        74: np.array((113, 83, 7)),
        75: np.array((69, 69, 69)),
        76: np.array((226, 79, 0)),
        77: np.array((181, 65, 41)),
        78: np.array((158, 97, 0)),
        79: np.array((113, 83, 7)),
        80: np.array((138, 138, 138)),
        81: np.array((93, 124, 200)),
        82: np.array((70, 156, 104)),
        83: np.array((26, 142, 166)),
        84: np.array((183, 152, 76)),
        85: np.array((138, 138, 138)),
        86: np.array((115, 171, 42)),
        87: np.array((70, 156, 104)),
        88: np.array((205, 120, 172)),
        89: np.array((161, 105, 234)),
        90: np.array((138, 138, 138)),
        91: np.array((93, 124, 200)),
        92: np.array((250, 134, 110)),
        93: np.array((205, 120, 172)),
        94: np.array((183, 152, 76)),
        95: np.array((138, 138, 138)),
        96: np.array((115, 171, 42)),
        97: np.array((70, 156, 104)),
        98: np.array((48, 189, 8)),
        99: np.array((3, 174, 70)),
        100: np.array((160, 185, 0)),
        101: np.array((115, 171, 42)),
        102: np.array((93, 203, 0)),
        103: np.array((48, 189, 8)),
        104: np.array((183, 152, 76)),
        105: np.array((138, 138, 138)),
        106: np.array((115, 171, 42)),
        107: np.array((70, 156, 104)),
        108: np.array((227, 167, 14)),
        109: np.array((183, 152, 76)),
        110: np.array((160, 185, 0)),
        111: np.array((115, 171, 42)),
        112: np.array((140, 225, 173)),
        113: np.array((95, 211, 235)),
        114: np.array((72, 244, 139)),
        115: np.array((28, 229, 201)),
        116: np.array((184, 240, 111)),
        117: np.array((140, 225, 173)),
        118: np.array((117, 255, 77)),
        119: np.array((72, 244, 139)),
        120: np.array((207, 207, 207)),
        121: np.array((162, 193, 255)),
        122: np.array((140, 225, 173)),
        123: np.array((95, 211, 235)),
        124: np.array((252, 221, 145)),
        125: np.array((207, 207, 207)),
        126: np.array((184, 240, 111)),
        127: np.array((140, 225, 173)),
        128: np.array((136, 50, 103)),
        129: np.array((91, 36, 165)),
        130: np.array((69, 69, 69)),
        131: np.array((24, 54, 131)),
        132: np.array((181, 65, 41)),
        133: np.array((136, 50, 103)),
        134: np.array((113, 83, 7)),
        135: np.array((69, 69, 69)),
        136: np.array((203, 32, 137)),
        137: np.array((159, 18, 199)),
        138: np.array((136, 50, 103)),
        139: np.array((91, 36, 165)),
        140: np.array((248, 47, 75)),
        141: np.array((203, 32, 137)),
        142: np.array((181, 65, 41)),
        143: np.array((136, 50, 103)),
        144: np.array((161, 105, 234)),
        145: np.array((116, 91, 255)),
        146: np.array((93, 124, 200)),
        147: np.array((48, 109, 255)),
        148: np.array((205, 120, 172)),
        149: np.array((161, 105, 234)),
        150: np.array((138, 138, 138)),
        151: np.array((93, 124, 200)),
        152: np.array((228, 87, 255)),
        153: np.array((183, 73, 255)),
        154: np.array((161, 105, 234)),
        155: np.array((116, 91, 255)),
        156: np.array((255, 101, 206)),
        157: np.array((228, 87, 255)),
        158: np.array((205, 120, 172)),
        159: np.array((161, 105, 234)),
        160: np.array((138, 138, 138)),
        161: np.array((93, 124, 200)),
        162: np.array((70, 156, 104)),
        163: np.array((26, 142, 166)),
        164: np.array((183, 152, 76)),
        165: np.array((138, 138, 138)),
        166: np.array((115, 171, 42)),
        167: np.array((70, 156, 104)),
        168: np.array((205, 120, 172)),
        169: np.array((161, 105, 234)),
        170: np.array((138, 138, 138)),
        171: np.array((93, 124, 200)),
        172: np.array((250, 134, 110)),
        173: np.array((205, 120, 172)),
        174: np.array((183, 152, 76)),
        175: np.array((138, 138, 138)),
        176: np.array((162, 193, 255)),
        177: np.array((118, 178, 255)),
        178: np.array((95, 211, 235)),
        179: np.array((50, 197, 255)),
        180: np.array((207, 207, 207)),
        181: np.array((162, 193, 255)),
        182: np.array((140, 225, 173)),
        183: np.array((95, 211, 235)),
        184: np.array((230, 174, 255)),
        185: np.array((185, 160, 255)),
        186: np.array((162, 193, 255)),
        187: np.array((118, 178, 255)),
        188: np.array((255, 189, 241)),
        189: np.array((230, 174, 255)),
        190: np.array((207, 207, 207)),
        191: np.array((162, 193, 255)),
        192: np.array((250, 134, 110)),
        193: np.array((205, 120, 172)),
        194: np.array((183, 152, 76)),
        195: np.array((138, 138, 138)),
        196: np.array((255, 148, 48)),
        197: np.array((250, 134, 110)),
        198: np.array((227, 167, 14)),
        199: np.array((183, 152, 76)),
        200: np.array((255, 116, 144)),
        201: np.array((255, 101, 206)),
        202: np.array((250, 134, 110)),
        203: np.array((205, 120, 172)),
        204: np.array((255, 130, 82)),
        205: np.array((255, 116, 144)),
        206: np.array((255, 148, 48)),
        207: np.array((250, 134, 110)),
        208: np.array((255, 189, 241)),
        209: np.array((230, 174, 255)),
        210: np.array((207, 207, 207)),
        211: np.array((162, 193, 255)),
        212: np.array((255, 203, 179)),
        213: np.array((255, 189, 241)),
        214: np.array((252, 221, 145)),
        215: np.array((207, 207, 207)),
        216: np.array((255, 171, 255)),
        217: np.array((255, 156, 255)),
        218: np.array((255, 189, 241)),
        219: np.array((230, 174, 255)),
        220: np.array((255, 185, 213)),
        221: np.array((255, 171, 255)),
        222: np.array((255, 203, 179)),
        223: np.array((255, 189, 241)),
        224: np.array((252, 221, 145)),
        225: np.array((207, 207, 207)),
        226: np.array((184, 240, 111)),
        227: np.array((140, 225, 173)),
        228: np.array((255, 236, 83)),
        229: np.array((252, 221, 145)),
        230: np.array((229, 254, 49)),
        231: np.array((184, 240, 111)),
        232: np.array((255, 203, 179)),
        233: np.array((255, 189, 241)),
        234: np.array((252, 221, 145)),
        235: np.array((207, 207, 207)),
        236: np.array((255, 218, 117)),
        237: np.array((255, 203, 179)),
        238: np.array((255, 236, 83)),
        239: np.array((252, 221, 145)),
        240: np.array((255, 255, 255)),
        241: np.array((232, 255, 255)),
        242: np.array((209, 255, 242)),
        243: np.array((164, 255, 255)),
        244: np.array((255, 255, 214)),
        245: np.array((255, 255, 255)),
        246: np.array((254, 255, 180)),
        247: np.array((209, 255, 242)),
        248: np.array((255, 255, 255)),
        249: np.array((255, 244, 255)),
        250: np.array((255, 255, 255)),
        251: np.array((232, 255, 255)),
        252: np.array((255, 255, 248)),
        253: np.array((255, 255, 255)),
        254: np.array((255, 255, 214)),
        255: np.array((255, 255, 255)),
    }
    # 84 unique colours


PALETTES = {
    'openemulator': OpenEmulatorPalette,
    'virtualii': VirtualIIPalette,
    'tohgr': ToHgrPalette,
    'ntsc': NTSCPalette
}

DEFAULT_PALETTE = 'openemulator'
