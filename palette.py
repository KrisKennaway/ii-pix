import numpy as np
import image


class Palette:
    RGB = None
    SRGB = None
    DOTS = None

    def __init__(self):
        # CIE2000 colour distance matrix from 24-bit RGB tuple to 4-bit
        # palette colour.
        self.distances = np.memmap("distances.npy", mode="r+",
                                   dtype=np.uint8, shape=(16777216, 16))

    # Default bmp2dhr palette
    sRGB = {
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
        RGB[k] = (np.clip(image.srgb_to_linear_array(v / 255), 0.0,
                          1.0) * 255).astype(
            np.uint8)
