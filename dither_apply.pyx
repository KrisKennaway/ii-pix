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
cdef apply_one_line(float[:, :, ::1] pattern, int el, int er, int xl, int xr, int y, float[:, ::1] image,
                   float[] quant_error):
    cdef int i, j
    cdef float *error = <float *> malloc(pattern.shape[1] * 3 * sizeof(float))

    for i in range(pattern.shape[1]):
        for j in range(3):
            error[i * 3 + j] = pattern[0, i, 0] * quant_error[j]

    for i in range(xr - xl):
        for j in range(3):
            image[xl+i, j] = clip(image[xl + i, j] + error[(el + i) * 3 + j], 0, 255)
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


cdef x_dither_bounds(float [:, :, ::1] pattern, int x_origin, int x_res, int x):
    cdef int el = max(x_origin - x, 0)
    cdef int er = min(pattern.shape[1], x_res - 1 - x)

    cdef int xl = x - x_origin + el
    cdef int xr = x - x_origin + er

    return el, er, xl, xr


def dither_lookahead(
        screen, float[:,:,::1] image_rgb, dither, differ, int x, int y, char[:, ::1] options_4bit,
        float[:, :, ::1] options_rgb, int lookahead):
    cdef float[:, :, ::1] pattern = dither.PATTERN
    cdef int x_res = screen.X_RES
    cdef int dither_x_origin = dither.ORIGIN[1]

    cdef int el, er, xl, xr
    el, er, xl, xr = x_dither_bounds(pattern, dither_x_origin, x_res, x)

    # X coord value of larger of dither bounding box or lookahead horizon
    cdef int xxr = min(max(x + lookahead, xr), x_res)

    # Copies of input pixels so we can dither in bulk
    # Leave enough space at right of image so we can dither the last of our lookahead pixels
    cdef float[:, :, ::1] lah_image_rgb = np.zeros(
        (2 ** lookahead, lookahead + xr - xl, 3), dtype=np.float32)
    lah_image_rgb[:, 0:xxr - x, :] = image_rgb[y, x:xxr, :]

    cdef float[:, ::] output_pixels
    cdef float *quant_error = <float *> malloc(2 ** lookahead * 3 * sizeof(float))

    cdef int i, j, k, l

    for i in range(xxr - x):
        # options_rgb choices are fixed, but we can still distribute
        # quantization error from having made these choices, in order to compute
        # the total error
        for k in range(2 ** lookahead):
            for l in range(3):
                quant_error[k * 3 + l] = lah_image_rgb[k, i, l] - options_rgb[k, i, l]

        # Don't update the input at position x (since we've already chosen
        # fixed outputs), but do propagate quantization errors to positions >x
        # so we can compensate for how good/bad these choices were
        el, er, xl, xr = x_dither_bounds(pattern, dither_x_origin, x_res, i)
        for j in range(2 ** lookahead):
            apply_one_line(pattern, el, er, xl, xr, 0, lah_image_rgb[j, :, :], &quant_error[j])

    free(quant_error)

    cdef int best
    cdef int best_error = 2**31-1
    cdef int total_error
    cdef long flat, dist, bit4

    cdef long r, g, b
    cdef (unsigned char)[:, ::1] distances = differ._distances
    for i in range(2**lookahead):
        total_error = 0
        for j in range(lookahead):
            # Clip lah_image_rgb into 0..255 range to prepare for computing colour distance
            r = long(clip(lah_image_rgb[i, j, 0], 0, 255))
            g = long(clip(lah_image_rgb[i, j, 1], 0, 255))
            b = long(clip(lah_image_rgb[i, j, 2], 0, 255))

            flat = (r << 16) + (g << 8) + b
            bit4 = options_4bit[i, j]
            dist = distances[flat, bit4]
            total_error += dist ** 2
            if total_error >= best_error:
                break
        if total_error < best_error:
            best_error = total_error
            best = i

    return options_4bit[best, 0], options_rgb[best, 0, :]