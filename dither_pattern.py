"""Error diffusion dither patterns."""

import numpy as np


class DitherPattern:
    PATTERN = None
    ORIGIN = None


class NoDither(DitherPattern):
    """No dithering."""
    PATTERN = np.array(((0, 0), (0, 0)),
                       dtype=np.float32).reshape(2, 2) / np.float(16)
    ORIGIN = (0, 1)


class FloydSteinbergDither(DitherPattern):
    """Floyd-Steinberg dither."""
    # 0 * 7
    # 3 5 1
    PATTERN = np.array(((0, 0, 7), (3, 5, 1)),
                       dtype=np.float32).reshape(2, 3) / np.float(16)
    ORIGIN = (0, 1)


class FloydSteinbergDither2(DitherPattern):
    """Floyd-Steinberg dither."""
    # 0 * 7
    # 3 5 1
    PATTERN = np.array(
        ((0, 0, 0, 0, 0, 7),
         (3, 5, 1, 0, 0, 0)),
        dtype=np.float32).reshape(2, 6) / np.float(16)
    ORIGIN = (0, 2)


class BuckelsDither(DitherPattern):
    """Default dither from bmp2dhr."""
    # 0 * 2 1
    # 1 2 1 0
    # 0 1 0 0
    PATTERN = np.array(((0, 0, 2, 1), (1, 2, 1, 0), (0, 1, 0, 0)),
                       dtype=np.float32).reshape(3, 4) / np.float32(8)
    ORIGIN = (0, 1)


class JarvisDither(DitherPattern):
    """Jarvis-Judice-Ninke dithering."""

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array(((0, 0, 0, 7, 5), (3, 5, 7, 5, 3), (1, 3, 5, 3, 1)),
                       dtype=np.float32).reshape(3, 5) / np.float32(48)
    ORIGIN = (0, 2)


class JarvisModifiedDither(DitherPattern):
    """Jarvis dithering, modified to diffuse errors to 4 forward x positions.

    This works well for double hi-res dithering, since the "best" colour
    match to a given pixel may only be accessible up to 4 x-positions further
    on.  Standard Jarvis dithering only propagates errors for 2 x-positions
    in the forward direction, which means that errors may have diffused away
    before we get to the pixel that can best take advantage of it.
    """

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array((
        (0, 0, 0, 15, 11, 7, 3),
        (3, 5, 7, 5, 3, 1, 0),
        (1, 3, 5, 3, 1, 0, 0)), dtype=np.float32).reshape(3, 7)
    PATTERN /= np.sum(PATTERN)
    ORIGIN = (0, 2)


PATTERNS = {
    'floyd': FloydSteinbergDither,
    'floyd2': FloydSteinbergDither2,
    'floyd-steinberg': FloydSteinbergDither,
    'buckels': BuckelsDither,
    'jarvis': JarvisDither,
    'jarvis-mod': JarvisModifiedDither,
    'none': NoDither
}

DEFAULT_PATTERN = 'floyd'
