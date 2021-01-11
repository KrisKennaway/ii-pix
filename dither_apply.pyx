# cython: infer_types=True

cimport cython
import numpy as np
# from cython.parallel import prange
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free


cdef float clip(float a, float min_value, float max_value) nogil:
    return min(max(a, min_value), max_value)


#@cython.boundscheck(False)
#@cython.wraparound(False)
def apply_one_line(float[:, :, ::1] pattern, int el, int er, int xl, int xr, int y, float[:, ::1] image,
                   float[::1] quant_error):
    cdef int i, j
    cdef float *error = <float *> malloc(pattern.shape[1] * quant_error.shape[0] * sizeof(float))

    #cdef float[:, ::1] error = cvarray(
    #    shape=(pattern.shape[1], quant_error.shape[0]), itemsize=sizeof(float), format="f")

    for i in range(pattern.shape[1]):
        for j in range(quant_error.shape[0]):
            error[i * quant_error.shape[0] + j] = pattern[0, i, 0] * quant_error[j]

    for i in range(xr - xl):
        for j in range(3):
            image[xl+i, j] = clip(image[xl + i, j] + error[(el + i) * quant_error.shape[0] + j], 0, 255)
    free(error)


# XXX cythonize
def apply(pattern, int el, int er, int xl, int xr, int et, int eb, int yt, int yb, image, quant_error):
    error = pattern * quant_error.reshape((1, 1, 3))

    # We could avoid clipping here, i.e. allow RGB values to extend beyond
    # 0..255 to capture a larger range of residual error.  This is faster
    # but seems to reduce image quality.
    # XXX extend image region to avoid need for boundary box clipping
    image[yt:yb, xl:xr, :] = np.clip(
        image[yt:yb, xl:xr, :] + error[et:eb, el:er, :], 0, 255)


def x_dither_bounds(dither, screen, int x):
    cdef int el = max(dither.ORIGIN[1] - x, 0)
    cdef int er = min(dither.PATTERN.shape[1], screen.X_RES - 1 - x)

    cdef int xl = x - dither.ORIGIN[1] + el
    cdef int xr = x - dither.ORIGIN[1] + er

    return el, er, xl, xr


def y_dither_bounds(dither, screen, int y):
    cdef int et = max(dither.ORIGIN[0] - y, 0)
    cdef int eb = min(dither.PATTERN.shape[0], screen.Y_RES - 1 - y)

    cdef int yt = y - dither.ORIGIN[0] + et
    cdef int yb = y - dither.ORIGIN[0] + eb

    return et, eb, yt, yb


def dither_lookahead(
        screen, float[:,:,::1] image_rgb, dither, differ, int x, int y, char[:, ::1] options_4bit,
        float[:, :, ::1] options_rgb, int lookahead):
    el, er, xl, xr = x_dither_bounds(dither, screen, x)

    # X coord value of larger of dither bounding box or lookahead horizon
    xxr = min(max(x + lookahead, xr), screen.X_RES)

    # copies of input pixels so we can dither in bulk
    # Leave enough space so we can dither the last of our lookahead pixels
    lah_image_rgb = np.zeros(
        (2 ** lookahead, lookahead + xr - xl, 3), dtype=np.float32)
    lah_image_rgb[:, 0:xxr - x, :] = np.copy(image_rgb[y, x:xxr, :])

    #cdef float[:, :, ::1] lah_image_rgb_view = lah_image_rgb
    #cdef float[:, :, ::1] options_rgb_view = options_rgb

    cdef float[:, ::] output_pixels
    cdef float[:, ::1] quant_error

    cdef int i, j
    for i in range(xxr - x):
        # options_rgb choices are fixed, but we can still distribute
        # quantization error from having made these choices, in order to compute
        # the total error
        input_pixels = np.copy(lah_image_rgb[:, i, :])
        output_pixels = options_rgb[:, i, :]
        quant_error = input_pixels - output_pixels
        # Don't update the input at position x (since we've already chosen
        # fixed outputs), but do propagate quantization errors to positions >x
        # so we can compensate for how good/bad these choices were
        el, er, xl, xr = x_dither_bounds(dither, screen, i)
        for j in range(2 ** lookahead):
            apply_one_line(dither.PATTERN, el, er, xl, xr, 0, lah_image_rgb[j, :, :], quant_error[j])

    error = differ.distance(np.clip(
        lah_image_rgb[:, 0:lookahead, :], 0, 255), options_4bit)
    total_error = np.sum(np.power(error, 2), axis=1)
    best = np.argmin(total_error)
    return options_4bit[best, 0], options_rgb[best, 0, :]