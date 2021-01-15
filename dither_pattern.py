import numpy as np


class DitherPattern:
    PATTERN = None
    ORIGIN = None


class FloydSteinbergDither(DitherPattern):
    # 0 * 7
    # 3 5 1
    PATTERN = np.array(((0, 0, 7), (3, 5, 1)),
                       dtype=np.float32).reshape(2, 3, 1) / np.float(16)
    # XXX X_ORIGIN since ORIGIN[0] == 0
    ORIGIN = (0, 1)


class BuckelsDither(DitherPattern):
    # 0 * 2 1
    # 1 2 1 0
    # 0 1 0 0
    PATTERN = np.array(((0, 0, 2, 1), (1, 2, 1, 0), (0, 1, 0, 0)),
                       dtype=np.float32).reshape(3, 4, 1) / np.float32(8)
    ORIGIN = (0, 1)


class JarvisDither(DitherPattern):
    # 0 0 X 7 5
    # 3 5 7 5 3
    # 1 3 5 3 1
    PATTERN = np.array(((0, 0, 0, 7, 5), (3, 5, 7, 5, 3), (1, 3, 5, 3, 1)),
                       dtype=np.float32).reshape(3, 5, 1) / np.float32(48)
    ORIGIN = (0, 2)

