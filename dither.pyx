# cython: infer_types=True

cimport cython
import numpy as np
# from cython.parallel import prange
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float clip(float a, float min_value, float max_value) nogil:
    return min(max(a, min_value), max_value)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void apply_one_line(float[:, :, ::1] pattern, int xl, int xr, float[] image, int image_shape0, float[] quant_error) nogil:
    cdef int i, j
    cdef float error

    for i in range(xr - xl):
        for j in range(3):
            error = pattern[0, i, 0] * quant_error[j]
            image[(xl + i) * image_shape0 + j] = clip(image[(xl + i) * image_shape0 + j] + error, 0, 255)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef apply(dither, screen, int x, int y, float [:, :, ::1]image, float[] quant_error):
    cdef int i, j, k

    # XXX only need 2 dimensions now
    cdef float[:, :, ::1] pattern = dither.PATTERN

    cdef int yt = dither_bounds_yt(dither.ORIGIN[0], y)
    cdef int yb = dither_bounds_yb(pattern, dither.ORIGIN[0], screen.Y_RES, y)
    cdef int xl = dither_bounds_xl(dither.ORIGIN[1], x)
    cdef int xr = dither_bounds_xr(pattern, dither.ORIGIN[1], screen.X_RES, x)

    cdef float error
    # We could avoid clipping here, i.e. allow RGB values to extend beyond
    # 0..255 to capture a larger range of residual error.  This is faster
    # but seems to reduce image quality.
    for i in range(yb - yt):
        for j in range(xr - xl):
            for k in range(3):
                error = pattern[i, j, 0] * quant_error[k]
                image[yt+i, xl+j, k] = clip(image[yt+i, xl+j, k] + error, 0, 255)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int dither_bounds_xl(int x_origin, int x):
    cdef int el = max(x_origin - x, 0)
    cdef int xl = x - x_origin + el
    return xl

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int dither_bounds_xr(float [:, :, ::1] pattern, int x_origin, int x_res, int x):
    cdef int er = min(pattern.shape[1], x_res - 1 - x)
    cdef int xr = x - x_origin + er
    return xr

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int dither_bounds_yt(int y_origin, int y):
    cdef int et = max(y_origin - y, 0)
    cdef int yt = y - y_origin + et

    return yt

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int dither_bounds_yb(float [:, :, ::1] pattern, int y_origin, int y_res, int y):
    cdef int eb = min(pattern.shape[0], y_res - 1 - y)
    cdef int yb = y - y_origin + eb
    return yb

@cython.boundscheck(False)
@cython.wraparound(False)
def dither_lookahead(
        screen, float[:,:,::1] image_rgb, dither, int x, int y, char[:, ::1] options_4bit,
        float[:, :, ::1] options_rgb, int lookahead):
    cdef float[:, :, ::1] pattern = dither.PATTERN
    cdef int x_res = screen.X_RES
    cdef int dither_x_origin = dither.ORIGIN[1]

    cdef int xl = dither_bounds_xl(dither_x_origin, x)
    cdef int xr = dither_bounds_xr(pattern, dither_x_origin, x_res, x)

    # X coord value of larger of dither bounding box or lookahead horizon
    cdef int xxr = min(max(x + lookahead, xr), x_res)

    cdef int i, j, k, l

    cdef int lah_shape0 = 2 ** lookahead
    cdef int lah_shape1 = lookahead + xr - xl
    cdef int lah_shape2 = 3
    cdef float *lah_image_rgb = <float *> malloc(lah_shape0 * lah_shape1 * lah_shape2 * sizeof(float))
    for i in range(2**lookahead):
        # Copies of input pixels so we can dither in bulk
        for j in range(xxr - x):
            for k in range(3):
                lah_image_rgb[i * lah_shape1 * lah_shape2 + j * lah_shape2 + k] = image_rgb[y, x+j, k]
        # Leave enough space at right of image so we can dither the last of our lookahead pixels.
        for j in range(xxr - x, lookahead + xr - xl):
            for k in range(3):
                lah_image_rgb[i * lah_shape1 * lah_shape2 + j * lah_shape2 + k] = 0

    cdef float[3] quant_error
    # Iterating by row then column is faster for some reason?
    for i in range(xxr - x):
        xl = dither_bounds_xl(dither_x_origin, i)
        xr = dither_bounds_xr(pattern, dither_x_origin, x_res, i)
        for j in range(2 ** lookahead):
            # Don't update the input at position x (since we've already chosen
            # fixed outputs), but do propagate quantization errors to positions >x
            # so we can compensate for how good/bad these choices were

            # options_rgb choices are fixed, but we can still distribute
            # quantization error from having made these choices, in order to compute
            # the total error
            for k in range(3):
                quant_error[k] = lah_image_rgb[j * lah_shape1 * lah_shape2 + i * lah_shape2 + k] - options_rgb[j, i, k]
            apply_one_line(pattern, xl, xr, &lah_image_rgb[j * lah_shape1 * lah_shape2], lah_shape2, quant_error)

    cdef int best
    cdef int best_error = 2**31-1
    cdef int total_error
    cdef long flat, dist, bit4

    cdef long r, g, b
    cdef (unsigned char)[:, ::1] distances = screen.palette.distances
    for i in range(2**lookahead):
        total_error = 0
        for j in range(lookahead):
            # Clip lah_image_rgb into 0..255 range to prepare for computing colour distance
            r = <long>clip(lah_image_rgb[i * lah_shape1 * lah_shape2 + j * lah_shape2 + 0], 0, 255)
            g = <long>clip(lah_image_rgb[i * lah_shape1 * lah_shape2 + j * lah_shape2 + 1], 0, 255)
            b = <long>clip(lah_image_rgb[i * lah_shape1 * lah_shape2 + j * lah_shape2 + 2], 0, 255)

            flat = (r << 16) + (g << 8) + b
            bit4 = options_4bit[i, j]
            dist = distances[flat, bit4]
            total_error += dist ** 2
            if total_error >= best_error:
                break
        if total_error < best_error:
            best_error = total_error
            best = i
    free(lah_image_rgb)
    return options_4bit[best, 0], options_rgb[best, 0, :]

import functools


@functools.lru_cache(None)
def lookahead_options(screen, lookahead, last_pixel_4bit, x):
    options_4bit = np.empty((2 ** lookahead, lookahead), dtype=np.uint8)
    options_rgb = np.empty((2 ** lookahead, lookahead, 3), dtype=np.float32)
    for i in range(2 ** lookahead):
        output_pixel_4bit = last_pixel_4bit
        for j in range(lookahead):
            xx = x + j
            palette_choices_4bit, palette_choices_rgb = \
                screen.pixel_palette_options(output_pixel_4bit, xx)
            output_pixel_4bit = palette_choices_4bit[(i & (1 << j)) >> j]
            output_pixel_rgb = np.array(
                palette_choices_rgb[(i & (1 << j)) >> j])
            options_4bit[i, j] = output_pixel_4bit
            options_rgb[i, j, :] = output_pixel_rgb

    return options_4bit, options_rgb


@cython.boundscheck(False)
@cython.wraparound(False)
def dither_image(
        screen, float[:, :, ::1] image_rgb, dither, int lookahead):
    cdef (unsigned char)[:, ::1] image_4bit = np.empty(
        (image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

    cdef int yres = screen.Y_RES
    cdef int xres = screen.X_RES

    cdef int y, x, i
    cdef float[3] quant_error
    cdef (unsigned char)[:, ::1] options_4bit
    cdef float[:, :, ::1] options_rgb
    cdef unsigned char output_pixel_4bit
    cdef float[::1] input_pixel_rgb

    for y in range(yres):
        # print(y)
        output_pixel_4bit = 0
        for x in range(xres):
            input_pixel_rgb = image_rgb[y, x, :]
            options_4bit, options_rgb = lookahead_options(
                screen, lookahead, output_pixel_4bit, x % 4)

            output_pixel_4bit, output_pixel_rgb = \
                dither_lookahead(
                    screen, image_rgb, dither, x, y, options_4bit,
                    options_rgb, lookahead)
            for i in range(3):
                quant_error[i] = input_pixel_rgb[i] - output_pixel_rgb[i]
                image_rgb[y, x, i] = output_pixel_rgb[i]
            image_4bit[y, x] = output_pixel_4bit
            apply(dither, screen, x, y, image_rgb, quant_error)

    return image_4bit, np.array(image_rgb)
