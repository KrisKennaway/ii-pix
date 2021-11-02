"""Representation of Apple II screen memory."""

import numpy as np
import palette as palette_py


class DHGRScreen:
    X_RES = 560
    Y_RES = 192

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

    def pack(self, bitmap: np.ndarray):
        """Packs an image into memory format (8k AUX + 8K MAIN)."""
        # The DHGR display encodes 7 pixels across interleaved 4-byte sequences
        # of AUX and MAIN memory, as follows:
        # PBBBAAAA PDDCCCCB PFEEEEDD PGGGGFFF
        # Aux N    Main N   Aux N+1  Main N+1  (N even)
        main_col = np.zeros(
            (self.Y_RES, self.X_RES // 14), dtype=np.uint8)
        aux_col = np.zeros(
            (self.Y_RES, self.X_RES // 14), dtype=np.uint8)
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
        return

    def bitmap_to_image_rgb(self, bitmap: np.ndarray) -> np.ndarray:
        """Convert our 2-bit bitmap image into a RGB image.

        Colour at every pixel is determined by the value of a 4-bit sliding
        window indexed by x % 4, which gives the index into our 16-colour RGB
        palette.
        """
        image_rgb = np.empty((self.Y_RES, self.X_RES, 3),
                             dtype=np.uint8)
        for y in range(self.Y_RES):
            pixel = [False, False, False, False]
            for x in range(self.X_RES):
                pixel[x % 4] = bitmap[y, x]
                dots = self.palette.DOTS_TO_INDEX[tuple(pixel)]
                image_rgb[y, x, :] = self.palette.RGB[dots]
        return image_rgb


# TODO: refactor to share implementation with DHGR560Screen
class DHGR560NTSCScreen(DHGRScreen):
    """DHGR screen including colour fringing and 8 pixel chroma bleed."""

    # TODO: unify with parent
    def bitmap_to_image_rgb(self, bitmap: np.ndarray) -> np.ndarray:
        """Convert our 2-bit bitmap image into a RGB image.

        Colour at every pixel is determined by the value of a 8-bit sliding
        window indexed by x % 4, which gives the index into our 256-colour RGB
        palette.
        """
        image_rgb = np.empty((self.Y_RES, self.X_RES, 3),
                             dtype=np.uint8)
        for y in range(self.Y_RES):
            pixel = [False, False, False, False, False, False, False, False]
            for x in range(self.X_RES):
                pixel = pixel[1:] + [bitmap[y, x]]
                dots = self.palette.DOTS_TO_INDEX[tuple(pixel)]
                image_rgb[y, x, :] = self.palette.RGB[dots, x % 4]
        return image_rgb

    @staticmethod
    def _sin(pos, phase0=0):
        x = pos % 12 + phase0
        return np.sin(x * 2 * np.pi / 12)

    @staticmethod
    def _cos(pos, phase0=0):
        x = pos % 12 + phase0
        return np.cos(x * 2 * np.pi / 12)

    def _read(self, line, pos):
        if pos < 0:
            return 0

        return 1 if line[pos] else 0

    def bitmap_to_ntsc(self, bitmap: np.ndarray) -> np.ndarray:
        y_width = 12
        u_width = 24
        v_width = 24

        contrast = 1
        # TODO: where does this come from?  OpenEmulator looks like it should
        #  use a value of 1.0 by default.
        saturation = 2
        # Fudge factor to make colours line up with OpenEmulator
        # TODO: where does this come from - is it due to the band-pass
        #  filtering they do?
        hue = -0.3

        # Apply effect of saturation
        yuv_to_rgb = np.array(
            ((1, 0, 0), (0, saturation, 0), (0, 0, saturation)), dtype=np.float)
        # Apply hue phase rotation
        yuv_to_rgb = np.matmul(np.array(
            ((1, 0, 0), (0, np.cos(hue), np.sin(hue)), (0, -np.sin(hue),
                                                        np.cos(hue)))),
            yuv_to_rgb)
        # Y'UV to R'G'B' conversion
        yuv_to_rgb = np.matmul(np.array(
            ((1, 0, 1.13983), (1, -0.39465, -.58060), (1, 2.03211, 0))),
            yuv_to_rgb)
        # Apply effect of contrast
        yuv_to_rgb *= contrast

        out_rgb = np.empty((bitmap.shape[0], bitmap.shape[1] * 3, 3),
                           dtype=np.uint8)
        for y in range(bitmap.shape[0]):
            ysum = 0
            usum = 0
            vsum = 0
            line = np.repeat(bitmap[y], 3)

            for x in range(bitmap.shape[1] * 3):
                ysum += self._read(line, x) - self._read(line, x - y_width)
                usum += self._read(line, x) * self._sin(x) - self._read(
                    line, x - u_width) * self._sin((x - u_width))
                vsum += self._read(line, x) * self._cos(x) - self._read(
                    line, x - v_width) * self._cos((x - v_width))
                rgb = np.matmul(
                    yuv_to_rgb, np.array(
                        (ysum / y_width, usum / u_width,
                         vsum / v_width)).reshape((3, 1))).reshape(3)
                r = min(255, max(0, rgb[0] * 255))
                g = min(255, max(0, rgb[1] * 255))
                b = min(255, max(0, rgb[2] * 255))
                out_rgb[y, x, :] = (r, g, b)

        return out_rgb
