# cython: infer_types=True

cimport cython
import numpy as np
# from cython.parallel import prange
from cython.view cimport array as cvarray


cdef float clip(float a, float min_value, float max_value) nogil:
    return min(max(a, min_value), max_value)


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_one_line(float[:, :, ::1] pattern, int el, int er, int xl, int xr, int y, float[:, :, ::1] image,
                   float[::1] quant_error):
    cdef int i, j
    cdef float[:, ::1] error = cvarray(
        shape=(pattern.shape[1], quant_error.shape[0]), itemsize=sizeof(float), format="f")

    for i in range(pattern.shape[1]):
        for j in range(quant_error.shape[0]):
            error[i, j] = pattern[0, i, 0] * quant_error[j]

    for i in range(xr - xl):
        for j in range(3):
            image[y, xl+i, j] = clip(image[y, xl + i, j] + error[el + i, j], 0, 255)


# XXX cythonize
def apply(pattern, int el, int er, int xl, int xr, int et, int eb, int yt, int yb, image, quant_error):
    error = pattern * quant_error.reshape((1, 1, 3))

    # We could avoid clipping here, i.e. allow RGB values to extend beyond
    # 0..255 to capture a larger range of residual error.  This is faster
    # but seems to reduce image quality.
    # XXX extend image region to avoid need for boundary box clipping
    image[yt:yb, xl:xr, :] = np.clip(
        image[yt:yb, xl:xr, :] + error[et:eb, el:er, :], 0, 255)
