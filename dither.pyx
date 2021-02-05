# cython: infer_types=True
# cython: profile=True

cimport cython
import functools
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free


# TODO: use a cdef class
cdef struct Dither:
    float* pattern   # Flattened dither pattern
    int x_shape
    int y_shape
    int x_origin
    int y_origin


cdef float clip(float a, float min_value, float max_value) nogil:
    return min(max(a, min_value), max_value)



cdef int dither_bounds_xl(Dither *dither, int x) nogil:
    cdef int el = max(dither.x_origin - x, 0)
    cdef int xl = x - dither.x_origin + el
    return xl


cdef int dither_bounds_xr(Dither *dither, int x_res, int x) nogil:
    cdef int er = min(dither.x_shape, x_res - x)
    cdef int xr = x - dither.x_origin + er
    return xr


cdef int dither_bounds_yt(Dither *dither, int y) nogil:
    cdef int et = max(dither.y_origin - y, 0)
    cdef int yt = y - dither.y_origin + et

    return yt

cdef int dither_bounds_yb(Dither *dither, int y_res, int y) nogil:
    cdef int eb = min(dither.y_shape, y_res - y)
    cdef int yb = y - dither.y_origin + eb
    return yb


# TODO: port screen.py to pyx
cdef unsigned char[::1] pixel_palette_options(object palette, int last_pixel_4bit, int x):
    # The two available colours for position x are given by the 4-bit
    # value of position x-1, and the 4-bit value produced by toggling the
    # value of the x % 4 bit (the current value of NTSC phase)
    last_dots = palette.DOTS[last_pixel_4bit]
    other_dots = list(last_dots)
    other_dots[x % 4] = not other_dots[x % 4]
    other_dots = tuple(other_dots)
    other_pixel_4bit = palette.DOTS_TO_4BIT[other_dots]

    cdef unsigned char res[2]
    res[0] = last_pixel_4bit
    res[1] = other_pixel_4bit

    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@functools.lru_cache(None)
def lookahead_options(object screen, int lookahead, unsigned char last_pixel_4bit, int x):
    cdef unsigned char[:, ::1] options_4bit = np.empty((2 ** lookahead, lookahead), dtype=np.uint8)
    # cdef float[:, :, ::1] options_rgb = np.empty((2 ** lookahead, lookahead, 3), dtype=np.float32)
    cdef int i, j, xx, p
    cdef unsigned char output_pixel_4bit
    #cdef unsigned char[::1] output_pixel_rgb

    cdef unsigned char[::1] palette_choices_4bit
    cdef unsigned char[:, ::1] palette_choices_rgb

    cdef object palette = screen.palette
    cdef dict palette_rgb = palette.RGB
    for i in range(2 ** lookahead):
        output_pixel_nbit = last_pixel_nbit
        for j in range(lookahead):
            xx = x + j
            palette_choices_4bit = pixel_palette_options(palette, output_pixel_4bit, xx)
            output_pixel_4bit = palette_choices_4bit[(i & (1 << j)) >> j]
            #output_pixel_rgb = palette_rgb[output_pixel_4bit]
            options_4bit[i, j] = output_pixel_4bit
            #for k in range(3):
            #    options_rgb[i, j, k] = <float>output_pixel_rgb[k]

    return options_4bit  # , options_rgb


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int dither_lookahead(Dither* dither, float[:, ::1] palette_rgb,
        float[:, :, ::1] image_rgb, int x, int y, unsigned char[:, ::1] options_4bit,
        # float[:, :, ::1] options_rgb,
        int lookahead, unsigned char[:, ::1] distances, int x_res):
    cdef int i, j, k, l

    # Don't bother dithering past the lookahead horizon or edge of screen.
    cdef int xxr = min(x + lookahead, x_res)
    cdef int lah_shape1 = xxr - x
    cdef int lah_shape2 = 3
    cdef float *lah_image_rgb = <float *> malloc(lah_shape1 * lah_shape2 * sizeof(float))

    cdef float[3] quant_error
    cdef unsigned char bit4
    cdef int best
    cdef int best_error = 2**31-1
    cdef int total_error
    cdef long flat, dist
    cdef long r, g, b

    for i in range(1 << lookahead):
        # Working copy of input pixels
        for j in range(xxr - x):
            for k in range(3):
                lah_image_rgb[j * lah_shape2 + k] = image_rgb[y, x+j, k]

        total_error = 0
        for j in range(xxr - x):
            xl = dither_bounds_xl(dither, j)
            xr = dither_bounds_xr(dither, xxr - x, j)
            # We don't update the input at position x (since we've already chosen
            # fixed outputs), but we do propagate quantization errors to positions >x
            # so we can compensate for how good/bad these choices were.  i.e. the
            # options_rgb choices are fixed, but we can still distribute quantization error
            # from having made these choices, in order to compute the total error.
            for k in range(3):
                quant_error[k] = lah_image_rgb[j * lah_shape2 + k] - palette_rgb[options_4bit[i,j], k]
            apply_one_line(dither, xl, xr, j, lah_image_rgb, lah_shape2, quant_error)

            r = <long>lah_image_rgb[j * lah_shape2 + 0]
            g = <long>lah_image_rgb[j * lah_shape2 + 1]
            b = <long>lah_image_rgb[j * lah_shape2 + 2]

            flat = (r << 16) + (g << 8) + b
            bit4 = options_nbit[i, j]
            dist = distances[flat, bit4]
            total_error += dist * dist
            if total_error >= best_error:
                break

        if total_error < best_error:
            best_error = total_error
            best = i
    free(lah_image_rgb)
    return best


cdef void apply_one_line(Dither* dither, int xl, int xr, int x, float[] image, int image_shape1, float[] quant_error) nogil:
    cdef int i, j
    cdef float error

    for i in range(xl, xr):
        for j in range(3):
            error = dither.pattern[i - x + dither.x_origin] * quant_error[j]
            image[i * image_shape1 + j] = clip(image[i * image_shape1 + j] + error, 0, 255)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void apply(Dither* dither, int x_res, int y_res, int x, int y, float[:,:,::1] image, float[] quant_error):
    cdef int i, j, k

    cdef int yt = dither_bounds_yt(dither, y)
    cdef int yb = dither_bounds_yb(dither, y_res, y)
    cdef int xl = dither_bounds_xl(dither, x)
    cdef int xr = dither_bounds_xr(dither, x_res, x)

    cdef float error
    # We could avoid clipping here, i.e. allow RGB values to extend beyond
    # 0..255 to capture a larger range of residual error.  This is faster
    # but seems to reduce image quality.
    # TODO: is this still true?
    for i in range(yt, yb):
        for j in range(xl, xr):
            for k in range(3):
                error = dither.pattern[(i - y) * dither.x_shape + j - x + dither.x_origin] * quant_error[k]
                image[i,j,k] = clip(image[i,j,k] + error, 0, 255)


@cython.boundscheck(False)
@cython.wraparound(False)
def find_nearest_colour(float[::1] pixel_rgb, unsigned char[::1] options_4bit, unsigned char[:, ::1] options_rgb, unsigned char[:, ::1] distances):
    cdef int best, dist
    cdef unsigned char bit4
    cdef int best_dist = 2**8
    cdef long flat

    for i in range(options_4bit.shape[0]):
        flat = (<long>pixel_rgb[0] << 16) + (<long>pixel_rgb[1] << 8) + <long>pixel_rgb[2]
        bit4 = options_nbit[i]
        dist = distances[flat, bit4]
        if dist < best_dist:
            best_dist = dist
            best = i

    return options_nbit[best], options_rgb[best, :]


@cython.boundscheck(False)
@cython.wraparound(False)
def dither_image(screen, float[:, :, ::1] image_rgb, dither, int lookahead, unsigned char verbose):
    cdef (unsigned char)[:, ::1] image_nbit = np.empty(
        (image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

    cdef int yres = screen.Y_RES
    cdef int xres = screen.X_RES

    cdef int y, x, i
    cdef float[3] input_pixel_rgb
    cdef float[3] quant_error
    cdef (unsigned char)[:, ::1] options_nbit
    cdef float[:, :, ::1] options_rgb
    cdef unsigned char [:, ::1] palette_choices_4bit
    cdef float[:, :, ::1] palette_choices_rgb
    cdef unsigned char output_pixel_4bit
    cdef float[::1] output_pixel_rgb

    # Flatten python dither pattern array for more efficient access
    # TODO: doesn't actually help?
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

    cdef (unsigned char)[:, ::1] distances = screen.palette.distances

    cdef float[:, ::1] palette_rgb = np.zeros((len(screen.palette.RGB), 3), dtype=np.float32)
    for i in screen.palette.RGB.keys():
        for j in range(3):
            palette_rgb[i, j] = screen.palette.RGB[i][j]

    for y in range(yres):
        if verbose:
            print("%d/%d" % (y, yres))
        output_pixel_nbit = 0
        for x in range(xres):
            for i in range(3):
                input_pixel_rgb[i] = image_rgb[y,x,i]
            if lookahead:
                palette_choices_4bit = lookahead_options(screen, lookahead, output_pixel_4bit, x % 4)
                best_idx = dither_lookahead(
                        &cdither, palette_rgb, image_rgb, x, y, palette_choices_4bit, lookahead, distances, xres)
                output_pixel_4bit = palette_choices_4bit[best_idx, 0]
                output_pixel_rgb = palette_rgb[output_pixel_4bit]
            #else:
            #    palette_choices_4bit, palette_choices_rgb = screen.pixel_palette_options(output_pixel_4bit, x)
            #    output_pixel_4bit, output_pixel_rgb = \
            #        find_nearest_colour(input_pixel_rgb, palette_choices_4bit, palette_choices_rgb, distances)
            for i in range(3):
                quant_error[i] = input_pixel_rgb[i] - output_pixel_rgb[i]
            image_4bit[y, x] = output_pixel_4bit
            apply(&cdither, xres, yres, x, y, image_rgb, quant_error)
            for i in range(3):
                image_rgb[y, x, i] = output_pixel_rgb[i]

    free(cdither.pattern)
    return image_4bit, np.array(image_rgb)
