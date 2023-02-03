"""Representation of Apple II screen memory."""

from enum import Enum
import numpy as np
import palette as palette_py


class Mode(Enum):
    LO_RES = 1
    DOUBLE_LO_RES = 2
    HI_RES = 3
    DOUBLE_HI_RES = 4
    SUPER_HI_RES_320 = 5
    SUPER_HI_RES_640 = 6
    SUPER_HI_RES_3200 = 7


class SHR320Screen:
    X_RES = 320
    Y_RES = 200

    MODE = Mode.SUPER_HI_RES_320

    def __init__(self):
        self.palettes = {k: np.zeros((16, 3), dtype=np.uint8) for k in
                         range(16)}
        # Really 4-bit values, indexing into palette
        self.pixels = np.array((self.Y_RES, self.X_RES), dtype=np.uint8)

        # Choice of palette per scan-line
        self.line_palette = np.zeros(self.Y_RES, dtype=np.uint8)

        self.memory = None

    def set_palette(self, idx: int, palette: np.array):
        if idx < 0 or idx > 15:
            raise ValueError("Palette index %s must be in range 0 .. 15" % idx)
        if palette.shape != (16, 3):
            raise ValueError("Palette size %s != (16, 3)" % palette.shape)
        # XXX check element range
        if palette.dtype != np.uint8:
            raise ValueError("Palette must be of type np.uint8")
        # print(palette)
        self.palettes[idx] = np.array(palette)

    def set_pixels(self, pixels):
        self.pixels = np.array(pixels)

    def pack(self):
        dump = np.zeros(32768, dtype=np.uint8)
        for y in range(self.Y_RES):
            pixel_pair = 0
            for x in range(self.X_RES):
                if x % 2 == 0:
                    pixel_pair |= (self.pixels[y, x] << 4)
                else:
                    pixel_pair |= self.pixels[y, x]
                    # print(pixel_pair)
                    dump[y * 160 + (x - 1) // 2] = pixel_pair
                    pixel_pair = 0

        scan_control_offset = 320 * 200 // 2
        for y in range(self.Y_RES):
            dump[scan_control_offset + y] = self.line_palette[y]

        palette_offset = scan_control_offset + 256
        for palette_idx, palette in self.palettes.items():
            for rgb_idx, rgb in enumerate(palette):
                r, g, b = rgb
                assert r <= 15 and g <= 15 and b <= 15
                # print(r, g, b)
                rgb_low = (g << 4) | b
                rgb_hi = r
                # print(hex(rgb_hi), hex(rgb_low))
                palette_idx_offset = palette_offset + (32 * palette_idx)
                dump[palette_idx_offset + (2 * rgb_idx)] = rgb_low
                dump[palette_idx_offset + (2 * rgb_idx + 1)] = rgb_hi

        self.memory = dump


class DHGRScreen:
    X_RES = 560
    Y_RES = 192

    MODE = Mode.DOUBLE_HI_RES

    def __init__(self):
        self.main = np.zeros(8192, dtype=np.uint8)
        self.aux = np.zeros(8192, dtype=np.uint8)

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


class NTSCScreen:
    NTSC_PHASE_SHIFT = None

    def _sin(self, pos):
        x = pos % 12 + self.NTSC_PHASE_SHIFT * 3
        return np.sin(x * 2 * np.pi / 12)

    def _cos(self, pos):
        x = pos % 12 + self.NTSC_PHASE_SHIFT * 3
        return np.cos(x * 2 * np.pi / 12)

    def _read(self, line, pos):
        if pos < 0:
            return 0
        return 1 if line[pos] else 0

    def bitmap_to_image_ntsc(self, bitmap: np.ndarray) -> np.ndarray:
        y_width = 12
        u_width = 24
        v_width = 24

        contrast = 1
        # TODO: This is necessary to match OpenEmulator.  I think it is because
        #  they introduce an extra (unexplained) factor of 2 when applying the
        #  Chebyshev/Lanczos filtering to the u and v components.
        saturation = 2
        # TODO: this phase shift is necessary to match OpenEmulator.  I'm not
        #  sure where it comes from - e.g. it doesn't match the phaseInfo
        #  calculation for the signal phase at the start of the visible region.
        hue = 0.2 * (2 * np.pi)

        # Apply effect of saturation
        yuv_to_rgb = np.array(
            ((1, 0, 0), (0, saturation, 0), (0, 0, saturation)),
            dtype=np.float32)
        # Apply hue phase rotation
        yuv_to_rgb = np.matmul(np.array(
            ((1, 0, 0), (0, np.cos(hue), np.sin(hue)), (0, -np.sin(hue),
                                                        np.cos(hue)))),
            yuv_to_rgb)
        # Y'UV to R'G'B' conversion
        yuv_to_rgb = np.matmul(np.array(
            ((1, 0, 1.139883), (1, -0.394642, -.5806227), (1, 2.032062, 0))),
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
                        # Mapping from bit pattern to colour is rotated by
                        # NTSC phase shift
                        np.roll(
                            np.array(bitmap_window, dtype=bool),
                            self.NTSC_PHASE_SHIFT
                        )
                    ), x % 4]
        return image_rgb


class DHGRNTSCScreen(DHGRScreen, NTSCScreen):
    def __init__(self, palette: palette_py.Palette):
        self.palette = palette
        super(DHGRNTSCScreen, self).__init__()

    NTSC_PHASE_SHIFT = 0


class HGRNTSCScreen(NTSCScreen):
    # Hi-Res really is 560 pixels horizontally, not 280 - but unlike DHGR
    # you can only independently control about half of the pixels.
    #
    # In more detail, hi-res graphics works like this:
    # - Each of the low 7 bits in a byte of screen memory results in
    #   enabling or disabling two sequential 560-resolution pixels.
    # - pixel screen order is from LSB to MSB
    # - if bit 8 (the "palette bit") is set then the 14-pixel sequence is
    #   shifted one position to the right, and the left-most pixel is filled
    #   in by duplicating the right-most pixel produced by the previous
    #   screen byte (i.e. bit 7)
    # - thus each byte produces a 15 or 14 pixel sequence depending on
    #   whether or not the palette bit is set.
    X_RES = 560
    Y_RES = 192

    MODE = Mode.HI_RES

    NTSC_PHASE_SHIFT = 3

    def __init__(self, palette: palette_py.Palette):
        self.main = np.zeros(8192, dtype=np.uint8)
        self.palette = palette
        super(HGRNTSCScreen, self).__init__()

    def pack_bytes(self, linear_bytemap: np.ndarray):
        """Packs an image into memory format (8K main)."""

        for y in range(self.Y_RES):
            addr = self.y_to_base_addr(y)
            self.main[addr:addr + 40] = linear_bytemap[y, :]
        return
