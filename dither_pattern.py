import numpy as np


class DitherPattern:
    PATTERN = None
    ORIGIN = None


class FloydSteinbergDither(DitherPattern):
    """Floyd-Steinberg dither."""
    # 0 * 7
    # 3 5 1
    PATTERN = np.array(((0, 0, 7), (3, 5, 1)),
                       dtype=np.float32).reshape(2, 3, 1) / np.float(16)
    # XXX X_ORIGIN since ORIGIN[0] == 0
    ORIGIN = (0, 1)


class FloydSteinbergDither2(DitherPattern):
    """Floyd-Steinberg dither."""
    # 0 * 7
    # 3 5 1
    PATTERN = np.array(
        ((0, 0, 0, 0, 0, 7),
         (3, 5, 1, 0, 0, 0)),
        dtype=np.float32).reshape(2, 6, 1) / np.float(16)
    # XXX X_ORIGIN since ORIGIN[0] == 0
    ORIGIN = (0, 2)


class BuckelsDither(DitherPattern):
    """Default dither from bmp2dhr."""
    # 0 * 2 1
    # 1 2 1 0
    # 0 1 0 0
    PATTERN = np.array(((0, 0, 2, 1), (1, 2, 1, 0), (0, 1, 0, 0)),
                       dtype=np.float32).reshape(3, 4, 1) / np.float32(8)
    ORIGIN = (0, 1)


class JarvisDither(DitherPattern):
    """Jarvis dithering."""

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array(((0, 0, 0, 7, 5), (3, 5, 7, 5, 3), (1, 3, 5, 3, 1)),
                       dtype=np.float32).reshape(3, 5, 1) / np.float32(48)
    ORIGIN = (0, 2)


class NoDither(DitherPattern):
    """Floyd-Steinberg dither."""
    # 0 * 7
    # 3 5 1
    PATTERN = np.array(((0, 0), (0, 0)),
                       dtype=np.float32).reshape(2, 2, 1) / np.float(16)
    # XXX X_ORIGIN since ORIGIN[0] == 0
    ORIGIN = (0, 1)


class TestDither(DitherPattern):
    """Jarvis dithering."""

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array((
        (0, 0, 0, 31, 29, 27, 25),
        (3, 5, 7, 5, 3, 1, 0),
        (1, 3, 5, 3, 1, 0, 0)), dtype=np.float32).reshape(3, 7, 1)
    PATTERN /= np.sum(PATTERN)
    ORIGIN = (0, 2)


class xTestDither(DitherPattern):
    """Jarvis dithering."""

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array((
        (0, 0, 15, 15, 15, 15),
        (3, 3, 5, 5, 1, 1)), dtype=np.float32).reshape(2, 6, 1)
    PATTERN /= np.sum(PATTERN)
    ORIGIN = (0, 1)


class TestDither(DitherPattern):
    """Jarvis dithering."""

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array((
        (0, 0, 0, 7, 7, 7, 7),
        (3, 5, 7, 5, 3, 1, 0),
        (1, 3, 5, 3, 1, 0, 0)), dtype=np.float32).reshape(3, 7, 1)
    PATTERN /= np.sum(PATTERN)
    ORIGIN = (0, 2)

# !!
class TestDither(DitherPattern):
    """Jarvis dithering."""

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array((
        (0, 0, 0, 9, 7, 5, 3),
        (3, 5, 7, 5, 3, 1, 0),
        (1, 3, 5, 3, 1, 0, 0)), dtype=np.float32).reshape(3, 7, 1)
    PATTERN /= np.sum(PATTERN)
    ORIGIN = (0, 2)

# !!!
class TestDither(DitherPattern):
    """Jarvis dithering."""

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array((
        (0, 0, 0, 15, 11, 7, 3),
        (3, 5, 7, 5, 3, 1, 0),
        (1, 3, 5, 3, 1, 0, 0)), dtype=np.float32).reshape(3, 7, 1)
    PATTERN /= np.sum(PATTERN)
    ORIGIN = (0, 2)

class xTestDither(DitherPattern):
    """Jarvis dithering."""

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array((
        (0, 0, 0, 9, 9,9,9),
        (3, 5, 7, 5, 3, 1, 0),
        (1, 3, 5, 3, 1, 0, 0)), dtype=np.float32).reshape(3, 7, 1)
    PATTERN /= np.sum(PATTERN)
    ORIGIN = (0, 2)
class xTestDither(DitherPattern):
    """Jarvis dithering."""

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array((
        (0, 0, 0, 15,13,11,9),
        (3, 5, 7, 5, 3, 1, 0),
        (1, 3, 5, 3, 1, 0, 0)), dtype=np.float32).reshape(3, 7, 1)
    PATTERN /= np.sum(PATTERN)
    ORIGIN = (0, 2)

class xTestDither(DitherPattern):
    """Jarvis dithering."""

    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array((
        (0, 0, 15,),
        (3, 5, 1,)), dtype=np.float32).reshape(2, 3, 1)
    PATTERN /= np.sum(PATTERN)
    ORIGIN = (0, 1)


PATTERNS = {
    'floyd': FloydSteinbergDither,
    'floyd2': FloydSteinbergDither2,
    'floyd-steinberg': FloydSteinbergDither,
    'buckels': BuckelsDither,
    'jarvis': JarvisDither,
    'test': TestDither,
    'none': NoDither
}

DEFAULT_PATTERN = 'jarvis'
