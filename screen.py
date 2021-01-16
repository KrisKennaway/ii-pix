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
        raise NotImplementedError

    def pack(self, image: np.ndarray):
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
        image_rgb = np.empty((192, 560, 3), dtype=np.uint8)
        for y in range(self.Y_RES):
            pixel = [False, False, False, False]
            for x in range(560):
                pixel[x % 4] = bitmap[y, x]
                dots = self.palette.DOTS_TO_4BIT[tuple(pixel)]
                image_rgb[y, x, :] = self.palette.RGB[dots]
        return image_rgb

    def pixel_palette_options(self, last_pixel_4bit, x: int):
        raise NotImplementedError


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
        last_dots = self.palette.DOTS[last_pixel_4bit]
        other_dots = list(last_dots)
        other_dots[x % 4] = not other_dots[x % 4]
        other_dots = tuple(other_dots)
        other_pixel_4bit = self.palette.DOTS_TO_4BIT[other_dots]
        return (
            np.array([last_pixel_4bit, other_pixel_4bit], dtype=np.uint8),
            np.array([self.palette.RGB[last_pixel_4bit],
                      self.palette.RGB[other_pixel_4bit]], dtype=np.uint8))
