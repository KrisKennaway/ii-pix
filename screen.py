"""Representation of Apple II screen memory."""

import math
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

        Colour at every pixel is determined by the value of an n-bit sliding
        window and x % 4, which give the index into our RGB palette.
        """
        image_rgb = np.empty((self.Y_RES, self.X_RES, 3), dtype=np.uint8)
        for y in range(self.Y_RES):
            bitmap_window = [False] * self.palette.PALETTE_DEPTH
            for x in range(self.X_RES):
                # Maintain a sliding window of pixels of width PALETTE_DEPTH
                bitmap_window = bitmap_window[1:] + [bitmap[y, x]]
                image_rgb[y, x, :] = self.palette.RGB[
                    self.palette.bitmap_to_idx(
                        np.array(bitmap_window, dtype=bool)), x % 4]
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

    def bitmap_to_image_ntsc(self, bitmap: np.ndarray) -> np.ndarray:
        y_width = 12
        i_width = 24
        q_width = 24

        contrast = 1
        saturation = 1
        # DHGR has a timing shift of 1/4 phase, i.e x=0 is actually 1/4 phase.
        # XXX should use (x + 1) % 4 ?
        hue = math.pi / 2

        # Apply effect of saturation
        yiq_to_rgb = np.array(
            ((1, 0, 0), (0, saturation, 0), (0, 0, saturation)), dtype=np.float)
        # Apply hue phase rotation
        yiq_to_rgb = np.matmul(np.array(
            ((1, 0, 0), (0, np.cos(hue), np.sin(hue)), (0, -np.sin(hue),
                                                        np.cos(hue)))),
            yiq_to_rgb)
        # Y'IQ to R'G'B' conversion
        yiq_to_rgb = np.matmul(np.array(
            ((1, 0.956, 0.621), (1, -0.272, -.647), (1, -1.107, 1.704))),
            yiq_to_rgb)

        # Apply effect of contrast
        yiq_to_rgb *= contrast

        out_rgb = np.empty((bitmap.shape[0], bitmap.shape[1] * 3, 3),
                           dtype=np.uint8)
        for y in range(bitmap.shape[0]):
            ysum = 0
            isum = 0
            qsum = 0
            line = np.repeat(bitmap[y], 3)

            for x in range(bitmap.shape[1] * 3):
                ysum += self._read(line, x) - self._read(line, x - y_width)
                isum += self._read(line, x) * self._sin(x) - self._read(
                    line, x - i_width) * self._sin((x - i_width))
                qsum += self._read(line, x) * self._cos(x) - self._read(
                    line, x - q_width) * self._cos((x - q_width))
                rgb = np.matmul(
                    yiq_to_rgb, np.array(
                        (ysum / y_width, isum / i_width,
                         qsum / q_width)).reshape((3, 1))).reshape(3)
                r = min(255, max(0, rgb[0] * 255))
                g = min(255, max(0, rgb[1] * 255))
                b = min(255, max(0, rgb[2] * 255))
                out_rgb[y, x, :] = (r, g, b)

        return out_rgb
