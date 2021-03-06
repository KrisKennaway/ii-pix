# cython: infer_types=True
# cython: profile=True

cimport cython
import functools
import numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free


# TODO: use a cdef class
# C representation of dither_pattern.DitherPattern data, for efficient access.
cdef struct Dither:

    float* pattern   # Flattened dither pattern
    int x_shape
    int y_shape
    int x_origin
    int y_origin


cdef float clip(float a, float min_value, float max_value) nogil:
    return min(max(a, min_value), max_value)


# Compute left-hand bounding box for dithering at horizontal position x.
cdef int dither_bounds_xl(Dither *dither, int x) nogil:
    cdef int el = max(dither.x_origin - x, 0)
    cdef int xl = x - dither.x_origin + el
    return xl


#Compute right-hand bounding box for dithering at horizontal position x.
cdef int dither_bounds_xr(Dither *dither, int x_res, int x) nogil:
    cdef int er = min(dither.x_shape, x_res - x)
    cdef int xr = x - dither.x_origin + er
    return xr


# Compute upper bounding box for dithering at vertical position y.
cdef int dither_bounds_yt(Dither *dither, int y) nogil:
    cdef int et = max(dither.y_origin - y, 0)
    cdef int yt = y - dither.y_origin + et

    return yt


# Compute lower bounding box for dithering at vertical position y.
cdef int dither_bounds_yb(Dither *dither, int y_res, int y) nogil:
    cdef int eb = min(dither.y_shape, y_res - y)
    cdef int yb = y - dither.y_origin + eb
    return yb


@cython.boundscheck(False)
@cython.wraparound(False)
@functools.lru_cache(None)
def lookahead_options(object screen, int lookahead, unsigned char last_pixel_nbit, int x):
    """Compute all possible n-bit palette values for upcoming pixels, given x coord and state of n pixels to the left.

    Args:
        screen: python screen.Screen object
        lookahead: how many pixels to lookahead
        last_pixel_nbit: n-bit value representing n pixels to left of current position, which determine available
            colours.
        x: current x position

    Returns: matrix of size (2**lookahead, lookahead) containing all 2**lookahead possible vectors of n-bit palette
        values accessible at positions x .. x + lookahead
    """
    cdef unsigned char[:, ::1] options_nbit = np.empty((2 ** lookahead, lookahead), dtype=np.uint8)
    cdef int i, j, xx, p
    cdef unsigned char output_pixel_nbit
    cdef unsigned char[::1] palette_choices_nbit

    cdef object palette = screen.palette
    cdef dict palette_rgb = palette.RGB
    for i in range(2 ** lookahead):
        output_pixel_nbit = last_pixel_nbit
        for j in range(lookahead):
            xx = x + j
            # Two possible n-bit palette choices at position xx, given state of n pixels to left.
            # TODO: port screen.py to pyx
            palette_choices_nbit = screen.pixel_palette_options(output_pixel_nbit, xx)
            output_pixel_nbit = palette_choices_nbit[(i & (1 << j)) >> j]
            options_nbit[i, j] = output_pixel_nbit

    return options_nbit


# Look ahead a number of pixels and compute choice for next pixel with lowest total squared error after dithering.
#
# Args:
#     dither: error diffusion pattern to apply
#     palette_rgb: matrix of all n-bit colour palette RGB values
#     image_rgb: RGB image in the process of dithering
#     x: current horizontal screen position
#     y: current vertical screen position
#     options_nbit: matrix of (2**lookahead, lookahead) possible n-bit colour choices at positions x .. x + lookahead
#     lookahead: how many horizontal pixels to look ahead
#     distances: matrix of (24-bit RGB, n-bit palette) perceptual colour distances
#     x_res: horizontal screen resolution
#
# Returns: index from 0 .. 2**lookahead into options_nbit representing best available choice for position (x,y)
#
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int dither_lookahead(Dither* dither, float[:, ::1] palette_rgb,
        float[:, :, ::1] image_rgb, int x, int y, unsigned char[:, ::1] options_nbit, int lookahead,
        unsigned char[:, ::1] distances, int x_res):
    cdef int i, j, k, l
    cdef float[3] quant_error
    cdef unsigned char bit4
    cdef int best
    cdef int best_error = 2**31-1
    cdef int total_error
    cdef long flat, dist
    cdef long r, g, b

    # Don't bother dithering past the lookahead horizon or edge of screen.
    cdef int xxr = min(x + lookahead, x_res)
    cdef int lah_shape1 = xxr - x
    cdef int lah_shape2 = 3
    cdef float *lah_image_rgb = <float *> malloc(lah_shape1 * lah_shape2 * sizeof(float))

    # For each 2**lookahead possibilities for the on/off state of the next lookahead pixels, apply error diffusion
    # and compute the total squared error to the source image.  Since we only have two possible colours for each
    # given pixel (dependent on the state already chosen for pixels to the left), we need to look beyond local minima.
    # i.e. it might be better to make a sub-optimal choice for this pixel if it allows access to much better pixel
    # colours at later positions.
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
                quant_error[k] = lah_image_rgb[j * lah_shape2 + k] - palette_rgb[options_nbit[i,j], k]
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


# Perform error diffusion to a single image row.
#
# Args:
#     dither: dither pattern to apply
#     xl: lower x bounding box
#     xr: upper x bounding box
#     x: starting horizontal position to apply error diffusion
#     image: array of shape (image_shape1, 3) representing RGB pixel data for a single image line, to be mutated.
#     image_shape1: horizontal dimension of image
#     quant_error: RGB quantization error to be diffused
#
cdef void apply_one_line(Dither* dither, int xl, int xr, int x, float[] image, int image_shape1,
        float[] quant_error) nogil:

    cdef int i, j
    cdef float error_fraction

    for i in range(xl, xr):
        error_fraction = dither.pattern[i - x + dither.x_origin]
        for j in range(3):
            image[i * image_shape1 + j] = clip(image[i * image_shape1 + j] + error_fraction * quant_error[j], 0, 255)


# Perform error diffusion across multiple image rows.
#
# Args:
#     dither: dither pattern to apply
#     x_res: horizontal image resolution
#     y_res: vertical image resolution
#     x: starting horizontal position to apply error diffusion
#     y: starting vertical position to apply error diffusion
#     image: RGB pixel data, to be mutated
#     quant_error: RGB quantization error to be diffused
#
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void apply(Dither* dither, int x_res, int y_res, int x, int y, float[:,:,::1] image, float[] quant_error):

    cdef int i, j, k

    cdef int yt = dither_bounds_yt(dither, y)
    cdef int yb = dither_bounds_yb(dither, y_res, y)
    cdef int xl = dither_bounds_xl(dither, x)
    cdef int xr = dither_bounds_xr(dither, x_res, x)

    cdef float error_fraction
    # We could avoid clipping here, i.e. allow RGB values to extend beyond
    # 0..255 to capture a larger range of residual error.  This is faster
    # but seems to reduce image quality.
    # TODO: is this still true?
    for i in range(yt, yb):
        for j in range(xl, xr):
            error_fraction = dither.pattern[(i - y) * dither.x_shape + j - x + dither.x_origin]
            for k in range(3):
                image[i,j,k] = clip(image[i,j,k] + error_fraction * quant_error[k], 0, 255)


# Compute closest colour from array of candidate n-bit colour palette values.
#
# Args:
#     pixel_rgb: source RGB colour value to be matched
#     options_nbit: array of candidate n-bit colour palette values
#     distances: matrix of (24-bit RGB value, n-bit colour value) perceptual colour differences
#
# Returns:
#     index of options_nbit entry having lowest distance value
#
@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned char find_nearest_colour(float[::1] pixel_rgb, unsigned char[::1] options_nbit,
        unsigned char[:, ::1] distances):

    cdef int best, dist
    cdef unsigned char bit4
    cdef int best_dist = 2**8
    cdef long flat

    for i in range(options_nbit.shape[0]):
        flat = (<long>pixel_rgb[0] << 16) + (<long>pixel_rgb[1] << 8) + <long>pixel_rgb[2]
        bit4 = options_nbit[i]
        dist = distances[flat, bit4]
        if dist < best_dist:
            best_dist = dist
            best = i

    return options_nbit[best]


# Dither a source image
#
# Args:
#     screen: screen.Screen object
#     image_rgb: input RGB image
#     dither: dither_pattern.DitherPattern to apply during dithering
#     lookahead: how many x positions to look ahead to optimize colour choices
#     verbose: whether to output progress during image conversion
#
# Returns: tuple of n-bit output image array and RGB output image array
#
@cython.boundscheck(False)
@cython.wraparound(False)
def dither_image(screen, float[:, :, ::1] image_rgb, dither, int lookahead, unsigned char verbose):
    cdef int y, x, i
    cdef float[3] input_pixel_rgb
    cdef float[3] quant_error
    cdef unsigned char [:, ::1] options_nbit
    cdef float[:, :, ::1] options_rgb
    cdef unsigned char [:, ::1] lookahead_palette_choices_nbit
    cdef unsigned char [::1] palette_choices_nbit
    cdef unsigned char output_pixel_nbit
    cdef float[::1] output_pixel_rgb

    # Hoist some python attribute accesses into C variables for efficient access during the main loop

    cdef int yres = screen.Y_RES
    cdef int xres = screen.X_RES

    cdef float[:, ::1] palette_rgb = np.zeros((len(screen.palette.RGB), 3), dtype=np.float32)
    for i in screen.palette.RGB.keys():
        for j in range(3):
            palette_rgb[i, j] = screen.palette.RGB[i][j]

    cdef (unsigned char)[:, ::1] distances = screen.palette.distances

    cdef Dither cdither
    cdither.y_shape = dither.PATTERN.shape[0]
    cdither.x_shape = dither.PATTERN.shape[1]
    cdither.y_origin = dither.ORIGIN[0]
    cdither.x_origin = dither.ORIGIN[1]
    # TODO: should be just as efficient to use a memoryview?
    cdither.pattern = <float *> malloc(cdither.x_shape * cdither.y_shape * sizeof(float))
    for i in range(cdither.y_shape):
        for j in range(cdither.x_shape):
            cdither.pattern[i * cdither.x_shape + j] = dither.PATTERN[i, j]

    cdef (unsigned char)[:, ::1] image_nbit = np.empty(
        (image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

    for y in range(yres):
        if verbose:
            print("%d/%d" % (y, yres))
        output_pixel_nbit = 0
        for x in range(xres):
            for i in range(3):
                input_pixel_rgb[i] = image_rgb[y,x,i]
            if lookahead:
                # Compute all possible 2**N choices of n-bit pixel colours for positions x .. x + lookahead
                lookahead_palette_choices_nbit = lookahead_options(screen, lookahead, output_pixel_nbit, x % 4)

                # Apply error diffusion for each of these 2**N choices, and compute which produces the closest match
                # to the source image over the succeeding N pixels
                best_idx = dither_lookahead(
                        &cdither, palette_rgb, image_rgb, x, y, lookahead_palette_choices_nbit, lookahead, distances,
                        xres)
                output_pixel_nbit = lookahead_palette_choices_nbit[best_idx, 0]
            else:
                # Choose the closest colour among the available n-bit palette options
                palette_choices_nbit = screen.pixel_palette_options(output_pixel_nbit, x)
                output_pixel_nbit = find_nearest_colour(input_pixel_rgb, palette_choices_nbit, distances)

            # Apply error diffusion from chosen output pixel value
            output_pixel_rgb = palette_rgb[output_pixel_nbit]
            for i in range(3):
                quant_error[i] = input_pixel_rgb[i] - output_pixel_rgb[i]
            image_nbit[y, x] = output_pixel_nbit
            apply(&cdither, xres, yres, x, y, image_rgb, quant_error)

            for i in range(3):
                image_rgb[y, x, i] = output_pixel_rgb[i]

    free(cdither.pattern)
    return image_nbit, np.array(image_rgb)
