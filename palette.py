"""RGB colour palettes to target for Apple II image conversions."""

import colour
import numpy as np
import image
import palette_ntsc


class Palette:
    # How many successive screen pixels are used to compute output pixel
    # palette index.
    PALETTE_DEPTH = None

    # These next three dictionaries are all indexed by a tuple of (n-bit pixel
    # value, NTSC phase), where:
    #   n == PALETTE_DEPTH
    #   MSB of the pixel value represents the current pixel on/off state
    #   LSB of the pixel value is the on/off state of the pixel n-1 positions
    #     to the left of current
    #   NTSC phase = 0 .. 3 (= x position % 4)
    #
    # The choice of LSB --> MSB increasing from left to right across the
    # screen matches the ordering used by the mapping of double hi-res memory
    # to screen pixels.
    #
    # Dictionary values are the colour of the corresponding pixel in various
    # colour spaces.

    # Values are pixel colour in sRGB colour space.  Palettes are defined in
    # this colour space.
    SRGB = None

    # Values are pixel colour in (linear) RGB colour space.  Dithering is
    # performed in this colour space.
    RGB = {}

    # Values are pixel colour in CAM16-UCS colour space.  This is used for
    # computing perceptual differences between colour values when optimizing
    # the image dithering.
    CAM16UCS = {}

    def __init__(self):
        self.RGB = {}
        for k, v in self.SRGB.items():
            self.RGB[k] = (np.clip(image.srgb_to_linear_array(v / 255), 0.0,
                                   1.0) * 255).astype(np.uint8)
            with colour.utilities.suppress_warnings(colour_usage_warnings=True):
                self.CAM16UCS[k] = colour.convert(
                    v / 255, "sRGB", "CAM16UCS").astype(np.float32)

    @staticmethod
    def _pixel_phase_shifts(phase_3_srgb):
        """Constructs dictionary of 4-bit pixel sequences for each NTSC phase.
        Assumes PALETTE_DEPTH == 4

        Args:
            phase_3_rgb: dict mapping 4-bit pixel sequence to sRGB values,
                for NTSC phase 3.

        Returns:
            dict mapping (shifted 4-bit pixel sequence, phase 0..3) to sRGB
            values
        """
        srgb_phases = {}
        for pixels, srgb in phase_3_srgb.items():
            srgb_phases[pixels, 3] = srgb
            # Rotate to compute 4-bit pixel sequences that produce the same
            # colour for NTSC phases 0..2
            for phase in range(0, 3):
                lsb = pixels & 1
                pixels >>= 1
                pixels |= lsb << 3
                srgb_phases[pixels, phase] = srgb
        return srgb_phases

    def bitmap_to_idx(self, pixels: np.array) -> int:
        """Converts a bitmap of pixels into integer representation.

        Args:
            pixels: 1-D array of booleans, representing a window of pixels from
              L to R.  Must be of size <= 8

        Returns:
            8-bit integer representation of pixels, suitable for use as an
            index into palette arrays
        """
        return np.packbits(
            # numpy uses big-endian representation which is the opposite
            # order to screen representation (i.e. LSB is the left-most
            # screen pixel), so we need to flip the order
            np.flip(pixels, axis=0)
        )[0] >> (8 - pixels.shape[0])


class ToHgrPalette(Palette):
    """4-bit palette used as default by other DHGR image converters."""
    PALETTE_DEPTH = 4

    # Default tohgr/bmp2dhr palette
    SRGB = Palette._pixel_phase_shifts({
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
    })


class OpenEmulatorPalette(Palette):
    """4-bit palette chosen to approximately match OpenEmulator output."""
    PALETTE_DEPTH = 4

    # OpenEmulator
    SRGB = Palette._pixel_phase_shifts({
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
    })


class VirtualIIPalette(Palette):
    """4-bit palette exactly matching Virtual II emulator output."""
    PALETTE_DEPTH = 4

    SRGB = Palette._pixel_phase_shifts({
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
    })


class NTSCPalette(Palette):
    """8-bit NTSC palette computed by averaging chroma signal over 8 pixels."""
    PALETTE_DEPTH = 8

    # Computed using ntsc_colours.py
    SRGB = palette_ntsc.SRGB


PALETTES = {
    'openemulator': OpenEmulatorPalette,
    'virtualii': VirtualIIPalette,
    'tohgr': ToHgrPalette,
    'ntsc': NTSCPalette
}

DEFAULT_PALETTE = 'ntsc'
