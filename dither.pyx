# cython: infer_types=True

cimport cython
import functools
import numpy as np
# from cython.parallel import prange
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free


# TODO: use a cdef class
cdef struct Dither:
    float* pattern   # Flattened dither pattern
    int x_shape
    int y_shape
    int x_origin
    int y_origin

cdef struct Image:
    float* flat   # Flattened image array
    int shape0
    int shape1
    int shape2


cdef float clip(float a, float min_value, float max_value) nogil:
    return min(max(a, min_value), max_value)



cdef int dither_bounds_xl(Dither *dither, int x):
    cdef int el = max(dither.x_origin - x, 0)
    cdef int xl = x - dither.x_origin + el
    return xl


cdef int dither_bounds_xr(Dither *dither, int x_res, int x):
    cdef int er = min(dither.x_shape, x_res - x)
    cdef int xr = x - dither.x_origin + er
    return xr


cdef int dither_bounds_yt(Dither *dither, int y):
    cdef int et = max(dither.y_origin - y, 0)
    cdef int yt = y - dither.y_origin + et

    return yt

cdef int dither_bounds_yb(Dither *dither, int y_res, int y):
    cdef int eb = min(dither.y_shape, y_res - y)
    cdef int yb = y - dither.y_origin + eb
    return yb


@cython.boundscheck(False)
@cython.wraparound(False)
@functools.lru_cache(None)
def lookahead_options(screen, lookahead, last_pixel_nbit, x):
    options_nbit = np.empty((2 ** lookahead, lookahead), dtype=np.uint8)
    options_rgb = np.empty((2 ** lookahead, lookahead, 3), dtype=np.float32)
    for i in range(2 ** lookahead):
        output_pixel_nbit = last_pixel_nbit
        for j in range(lookahead):
            xx = x + j
            palette_choices_nbit, palette_choices_rgb = \
                screen.pixel_palette_options(output_pixel_nbit, xx)
            output_pixel_nbit = palette_choices_nbit[(i & (1 << j)) >> j]
            output_pixel_rgb = np.array(
                palette_choices_rgb[(i & (1 << j)) >> j])
            options_nbit[i, j] = output_pixel_nbit
            options_rgb[i, j, :] = output_pixel_rgb

    return options_nbit, options_rgb


@cython.boundscheck(False)
@cython.wraparound(False)
cdef dither_lookahead(Dither* dither,
        screen, Image* image_rgb, int x, int y, unsigned char[:, ::1] options_nbit,
        float[:, :, ::1] options_rgb, int lookahead):
    cdef int x_res = screen.X_RES

    cdef int xl = dither_bounds_xl(dither, x)
    cdef int xr = dither_bounds_xr(dither, x_res, x)

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
                lah_image_rgb[i * lah_shape1 * lah_shape2 + j * lah_shape2 + k] = image_rgb.flat[
                    y * image_rgb.shape1 * image_rgb.shape2 + (x+j) * image_rgb.shape2 + k]
        # Leave enough space at right of image so we can dither the last of our lookahead pixels.
        for j in range(xxr - x, lookahead + xr - xl):
            for k in range(3):
                lah_image_rgb[i * lah_shape1 * lah_shape2 + j * lah_shape2 + k] = 0

    cdef float[3] quant_error
    # Iterating by row then column is faster for some reason?
    for i in range(xxr - x):
        xl = dither_bounds_xl(dither, i)
        xr = dither_bounds_xr(dither, x_res - x, i)
        for j in range(2 ** lookahead):
            # Don't update the input at position x (since we've already chosen
            # fixed outputs), but do propagate quantization errors to positions >x
            # so we can compensate for how good/bad these choices were

            # options_rgb choices are fixed, but we can still distribute
            # quantization error from having made these choices, in order to compute
            # the total error
            for k in range(3):
                quant_error[k] = lah_image_rgb[j * lah_shape1 * lah_shape2 + i * lah_shape2 + k] - options_rgb[j, i, k]
            apply_one_line(dither, xl, xr, i, &lah_image_rgb[j * lah_shape1 * lah_shape2], lah_shape2, quant_error)

    cdef unsigned char bit4
    cdef int best
    cdef int best_error = 2**31-1
    cdef int total_error
    cdef long flat, dist

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
            bit4 = options_nbit[i, j]
            dist = distances[flat, bit4]
            total_error += dist ** 2
            if total_error >= best_error:
                break
        if total_error < best_error:
            best_error = total_error
            best = i
    free(lah_image_rgb)
    return options_nbit[best, 0], options_rgb[best, 0, :]


cdef void apply_one_line(Dither* dither, int xl, int xr, int x, float[] image, int image_shape1, float[] quant_error):
    cdef int i, j
    cdef float error

    for i in range(xl, xr):
        for j in range(3):
            error = dither.pattern[i - x + dither.x_origin] * quant_error[j]
            image[i * image_shape1 + j] = clip(image[i * image_shape1 + j] + error, 0, 255)


cdef apply(Dither* dither, screen, int x, int y, Image* image, float[] quant_error):
    cdef int i, j, k

    cdef int yt = dither_bounds_yt(dither, y)
    cdef int yb = dither_bounds_yb(dither, screen.Y_RES, y)
    cdef int xl = dither_bounds_xl(dither, x)
    cdef int xr = dither_bounds_xr(dither, screen.X_RES, x)

    cdef float error
    # We could avoid clipping here, i.e. allow RGB values to extend beyond
    # 0..255 to capture a larger range of residual error.  This is faster
    # but seems to reduce image quality.
    # TODO: is this still true?
    for i in range(yt, yb):
        for j in range(xl, xr):
            for k in range(3):
                error = dither.pattern[(i - y) * dither.x_shape + j - x + dither.x_origin] * quant_error[k]
                image.flat[i * image.shape1 * image.shape2 + j * image.shape2 + k] = clip(
                    image.flat[i * image.shape1 * image.shape2 + j * image.shape2 + k] + error, 0, 255)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef find_nearest_colour(screen, float[3] pixel_rgb, unsigned char[::1] options_nbit, unsigned char[:, ::1] options_rgb):
    cdef int best, dist, i
    cdef unsigned char bit4
    cdef int best_dist = 2**8
    cdef long flat

    cdef (unsigned char)[:, ::1] distances = screen.palette.distances
    for i in range(options_nbit.shape[0]):
        flat = (<long>pixel_rgb[0] << 16) + (<long>pixel_rgb[1] << 8) + <long>pixel_rgb[2]
        bit4 = options_nbit[i]
        dist = distances[flat, bit4]
        if dist < best_dist:
            best_dist = dist
            best = i

    return options_nbit[best], options_rgb[best, :]


@cython.boundscheck(False)
@cython.wraparound(False)
def dither_image(screen, float[:, :, ::1] image_rgb, dither, int lookahead):
    cdef (unsigned char)[:, ::1] image_nbit = np.empty(
        (image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

    cdef int yres = screen.Y_RES
    cdef int xres = screen.X_RES

    cdef int y, x, i
    cdef float[3] quant_error
    cdef (unsigned char)[:, ::1] options_nbit
    cdef float[:, :, ::1] options_rgb
    cdef unsigned char output_pixel_nbit
    cdef float[3] input_pixel_rgb

    # Flatten python image array for more efficient access
    cdef Image cimage_rgb
    cimage_rgb.flat = <float *> malloc(image_rgb.shape[0] * image_rgb.shape[1] * image_rgb.shape[2] * sizeof(float))
    cimage_rgb.shape0 = image_rgb.shape[0]
    cimage_rgb.shape1 = image_rgb.shape[1]
    cimage_rgb.shape2 = image_rgb.shape[2]
    for y in range(cimage_rgb.shape0):
        for x in range(cimage_rgb.shape1):
            for i in range(cimage_rgb.shape2):
                cimage_rgb.flat[y * cimage_rgb.shape1 * cimage_rgb.shape2 + x * cimage_rgb.shape2 + i] = (
                    image_rgb[y, x, i])

    # Flatten python dither pattern array for more efficient access
    cdef Dither cdither
    cdither.y_shape = dither.PATTERN.shape[0]
    cdither.x_shape = dither.PATTERN.shape[1]
    cdither.y_origin = dither.ORIGIN[0]
    cdither.x_origin = dither.ORIGIN[1]
    # Convert dither.PATTERN to a malloced array which is faster to access
    cdither.pattern = <float *> malloc(cdither.x_shape * cdither.y_shape * sizeof(float))
    for i in range(cdither.y_shape):
        for j in range(cdither.x_shape):
            cdither.pattern[i * cdither.x_shape + j] = dither.PATTERN[i, j, 0]

    for y in range(yres):
        output_pixel_nbit = 0
        for x in range(xres):
            for i in range(3):
                input_pixel_rgb[i] = cimage_rgb.flat[
                    y * cimage_rgb.shape1 * cimage_rgb.shape2 + x * cimage_rgb.shape2 + i]
            if lookahead:
                palette_choices_nbit, palette_choices_rgb = lookahead_options(
                    screen, lookahead, output_pixel_nbit, x % 4)
                output_pixel_nbit, output_pixel_rgb = \
                    dither_lookahead(
                        &cdither, screen, &cimage_rgb, x, y, palette_choices_nbit, palette_choices_rgb, lookahead)
            else:
                palette_choices_nbit, palette_choices_rgb = screen.pixel_palette_options(output_pixel_nbit, x)
                output_pixel_nbit, output_pixel_rgb = \
                    find_nearest_colour(screen, input_pixel_rgb, palette_choices_nbit, palette_choices_rgb)
            for i in range(3):
                quant_error[i] = input_pixel_rgb[i] - output_pixel_rgb[i]
            image_nbit[y, x] = output_pixel_nbit
            apply(&cdither, screen, x, y, &cimage_rgb, quant_error)
            for i in range(3):
                image_rgb[y, x, i] = output_pixel_rgb[i]

    free(cdither.pattern)
    free(cimage_rgb.flat)
    return image_nbit, np.array(image_rgb)
