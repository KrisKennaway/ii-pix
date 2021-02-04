"""Representation of Apple II screen memory."""

import numpy as np
import palette as palette_py


class Screen:
    X_RES = None
    Y_RES = None
    X_PIXEL_WIDTH = None

    def __init__(self, palette: palette_py.Palette):
        self.main = np.zeros(8192, dtype=np.uint8)
        self.aux = np.zeros(8192, dtype=np.uint8)
        self.palette = palette

    @staticmethod
    def y_to_base_addr(y: int) -> int:
        """Maps y coordinate to screen memory base address."""
        a = y // 64
        d = y - 64 * a
        b = d // 8
        c = d - 8 * b

        return 1024 * c + 128 * b + 40 * a

    def _image_to_bitmap(self, image: np.ndarray) -> np.ndarray:
        """Converts 4-bit image to 2-bit image bitmap.

        Each 4-bit colour value maps to a sliding window of 4 successive pixels,
        via x%4.
        """
        raise NotImplementedError

    def pack(self, image: np.ndarray):
        """Packs an image into memory format (8k AUX + 8K MAIN)."""
        bitmap = self._image_to_bitmap(image)
        # The DHGR display encodes 7 pixels across interleaved 4-byte sequences
        # of AUX and MAIN memory, as follows:
        # PBBBAAAA PDDCCCCB PFEEEEDD PGGGGFFF
        # Aux N    Main N   Aux N+1  Main N+1  (N even)
        main_col = np.zeros(
            (self.Y_RES, self.X_RES * self.X_PIXEL_WIDTH // 14), dtype=np.uint8)
        aux_col = np.zeros(
            (self.Y_RES, self.X_RES * self.X_PIXEL_WIDTH // 14), dtype=np.uint8)
        for byte_offset in range(80):
            column = np.zeros(self.Y_RES, dtype=np.uint8)
            for bit in range(7):
                column |= (bitmap[:, 7 * byte_offset + bit].astype(
                    np.uint8) << bit)
            if byte_offset % 2 == 0:
                aux_col[:, byte_offset // 2] = column
            else:
                main_col[:, (byte_offset - 1) // 2] = column

        for y in range(self.Y_RES):
            addr = self.y_to_base_addr(y)
            self.aux[addr:addr + 40] = aux_col[y, :]
            self.main[addr:addr + 40] = main_col[y, :]

        return bitmap

    def bitmap_to_image_rgb(self, bitmap: np.ndarray) -> np.ndarray:
        """Convert our 2-bit bitmap image into a RGB image.

        Colour at every pixel is determined by the value of a 4-bit sliding
        window indexed by x % 4, which gives the index into our 16-colour RGB
        palette.
        """
        image_rgb = np.empty((192, 560, 3), dtype=np.uint8)
        for y in range(self.Y_RES):
            pixel = [False, False, False, False]
            for x in range(560):
                pixel[x % 4] = bitmap[y, x]
                dots = self.palette.DOTS_TO_4BIT[tuple(pixel)]
                image_rgb[y, x, :] = self.palette.RGB[dots]
        return image_rgb

    def pixel_palette_options(self, last_pixel_4bit, x: int):
        """Returns available colours for given x pos and 4-bit colour of x-1"""
        raise NotImplementedError

    @staticmethod
    def _sin(pos, phase0=3):
        x = pos % 12 + phase0
        return 8 * np.sin(x * 2 * np.pi / 12)

    @staticmethod
    def _cos(pos, phase0=3):
        x = pos % 12 + phase0
        return 8 * np.cos(x * 2 * np.pi / 12)

    def _read(self, line, pos):
        if pos < 0:
            return 0

        # Sather says black level is 0.36V and white level 1.1V, but this
        # doesn't seem to be right (they correspond to values -29 and +33)
        # which means that 0101 grey has Y value ~0, i.e. is black.  These are
        # only mentioned as labels on figure 8.2 though.
        #
        # _The Apple II Circuit description_ by W. Gayler gives black=0.5V
        # and white=2.0V which is much more plausible.
        #
        # Conversion is given by floor((voltage-0.518)*1000/12)-15
        return 108 if line[pos] else 0  # -16

    def bitmap_to_ntsc(self, bitmap: np.ndarray) -> np.ndarray:
        """
        See http://forums.nesdev.com/viewtopic.php?p=172329#p172329
        """
        y_width = 12
        i_width = 24
        q_width = 24

        contrast = 167941
        saturation = 144044

        yr = contrast / y_width
        ir = contrast * 1.994681e-6 * saturation / i_width
        qr = contrast * 9.915742e-7 * saturation / q_width

        yg = contrast / y_width
        ig = contrast * 9.151351e-8 * saturation / i_width
        qg = contrast * -6.334805e-7 * saturation / q_width

        yb = contrast / y_width
        ib = contrast * -1.012984e-6 * saturation / i_width
        qb = contrast * 1.667217e-6 * saturation / q_width

        out_rgb = np.empty((192, 560 * 3, 3), dtype=np.uint8)
        for y in range(self.Y_RES):
            ysum = 0
            isum = 0
            qsum = 0
            line = np.repeat(bitmap[y], 3)

            color = y // (192//16)
            line = np.repeat(np.tile((color & 1, color & 2, color & 4,
                                      color & 8), 140), 3)
            for x in range(560 * 3):
                ysum += self._read(line, x) - self._read(line, x - y_width)
                isum += self._read(line, x) * self._cos(x) - self._read(
                    line, x - i_width) * self._cos((x - i_width))
                qsum += self._read(line, x) * self._sin(x) - self._read(
                    line, x - q_width) * self._sin((x - q_width))

                r = min(255, max(0, ysum * yr + isum * ir + qsum * qr) /
                        65536)
                g = min(255,
                        max(0, (ysum * yg + isum * ig + qsum * qg) / 65536))
                b = min(255,
                        max(0, (ysum * yb + isum * ib + qsum * qb) / 65536))
                out_rgb[y, x, :] = (r, g, b)

        return out_rgb


class DHGR140Screen(Screen):
    """DHGR screen ignoring colour fringing, i.e. treating as 140x192x16."""

    X_RES = 140
    Y_RES = 192
    X_PIXEL_WIDTH = 4

    def _image_to_bitmap(self, image_4bit: np.ndarray) -> np.ndarray:
        bitmap = np.zeros(
            (self.Y_RES, self.X_RES * self.X_PIXEL_WIDTH), dtype=np.bool)
        for y in range(self.Y_RES):
            for x in range(self.X_RES):
                pixel = image_4bit[y, x]
                dots = self.palette.DOTS[pixel]
                bitmap[y, x * self.X_PIXEL_WIDTH:(
                        (x + 1) * self.X_PIXEL_WIDTH)] = dots
        return bitmap

    def pixel_palette_options(self, last_pixel_4bit, x: int):
        # All 16 colour choices are available at every x position.
        return (
            np.array(list(self.palette.RGB.keys()), dtype=np.uint8),
            np.array(list(self.palette.RGB.values()), dtype=np.uint8))


class DHGR560Screen(Screen):
    """DHGR screen including colour fringing."""
    X_RES = 560
    Y_RES = 192
    X_PIXEL_WIDTH = 1

    def _image_to_bitmap(self, image_4bit: np.ndarray) -> np.ndarray:
        bitmap = np.zeros((self.Y_RES, self.X_RES), dtype=np.bool)
        for y in range(self.Y_RES):
            for x in range(self.X_RES):
                pixel = image_4bit[y, x]
                dots = self.palette.DOTS[pixel]
                phase = x % 4
                bitmap[y, x] = dots[phase]
        return bitmap

    def pixel_palette_options(self, last_pixel_4bit, x: int):
        # The two available colours for position x are given by the 4-bit
        # value of position x-1, and the 4-bit value produced by toggling the
        # value of the x % 4 bit (the current value of NTSC phase)
        last_dots = self.palette.DOTS[last_pixel_4bit]
        other_dots = list(last_dots)
        other_dots[x % 4] = not other_dots[x % 4]
        other_dots = tuple(other_dots)
        other_pixel_4bit = self.palette.DOTS_TO_4BIT[other_dots]
        return (
            np.array([last_pixel_4bit, other_pixel_4bit], dtype=np.uint8),
            np.array([self.palette.RGB[last_pixel_4bit],
                      self.palette.RGB[other_pixel_4bit]], dtype=np.uint8))
