# cython: infer_types=True
# cython: profile=False

cimport cython
import colour
import math
import numpy as np
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


cdef inline unsigned char lookahead_pixels(unsigned char last_pixel_nbit, unsigned int next_pixels, int lookahead) nogil:
    """Compute all possible n-bit palette values for upcoming pixels, given x coord and state of n pixels to the left.

    Args:
        XXX
        screen: python screen.Screen object
        lookahead: how many pixels to lookahead
        last_pixel_nbit: n-bit value representing n pixels to left of current position, which determine available
            colours.
        x: current x position

    Returns: matrix of size (2**lookahead, lookahead) containing all 2**lookahead possible vectors of n-bit palette
        values accessible at positions x .. x + lookahead
    """
    # XXX palette bit depth
    return (last_pixel_nbit >> (lookahead+1)) | (next_pixels << (8 - (lookahead + 1)))


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
cdef int dither_lookahead(Dither* dither, float[:, :, ::1] palette_cam16, float[:, :, ::1] palette_rgb,
        float[:, :, ::1] image_rgb, int x, int y, int lookahead, unsigned char last_pixels,
        int x_res, float[:,::1] rgb_to_cam16ucs) nogil:
    cdef int i, j, k
    cdef float[3] quant_error
    cdef int best
    cdef float best_error = 2**31-1
    cdef float total_error
    cdef unsigned char next_pixels
    cdef int phase

    # Don't bother dithering past the lookahead horizon or edge of screen.
    cdef int xxr = min(x + lookahead, x_res)
    cdef int lah_shape1 = xxr - x
    cdef int lah_shape2 = 3
    # XXX use a memoryview
    cdef float *lah_image_rgb = <float *> malloc(lah_shape1 * lah_shape2 * sizeof(float))
    cdef float[::1] lah_cam16ucs

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
            phase = (x + j) % 4

            next_pixels = lookahead_pixels(last_pixels, next_pixels=i, lookahead=j)

            # We don't update the input at position x (since we've already chosen
            # fixed outputs), but we do propagate quantization errors to positions >x
            # so we can compensate for how good/bad these choices were.  i.e. the
            # options_rgb choices are fixed, but we can still distribute quantization error
            # from having made these choices, in order to compute the total error.
            for k in range(3):
                quant_error[k] = lah_image_rgb[j * lah_shape2 + k] - palette_rgb[next_pixels, phase, k]
            apply_one_line(dither, xl, xr, j, lah_image_rgb, lah_shape2, quant_error)

            lah_cam16ucs = convert_rgb_to_cam16ucs(
                rgb_to_cam16ucs, lah_image_rgb[j*lah_shape2], lah_image_rgb[j*lah_shape2+1], lah_image_rgb[j*lah_shape2+2])
            total_error += colour_distance_squared(lah_cam16ucs, palette_cam16[next_pixels, phase])

            if total_error >= best_error:
                break

        if total_error < best_error:
            best_error = total_error
            best = i

    free(lah_image_rgb)
    return best


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float[::1] convert_rgb_to_cam16ucs(float[:, ::1] rgb_to_cam16ucs, float r, float g, float b) nogil:
    cdef int rgb_24bit = (<int>(r*255) << 16) + (<int>(g*255) << 8) + <int>(b*255)
    return rgb_to_cam16ucs[rgb_24bit]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float colour_distance_squared(float[::1] colour1, float[::1] colour2) nogil:
    return (colour1[0] - colour2[0])**2 + (colour1[1] - colour2[1])**2 + (colour1[2] - colour2[2])**2


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
            image[i * image_shape1 + j] = clip(image[i * image_shape1 + j] + error_fraction * quant_error[j], 0, 1)


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
cdef void apply(Dither* dither, int x_res, int y_res, int x, int y, float[:,:,::1] image, float[] quant_error) nogil:

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
                image[i,j,k] = clip(image[i,j,k] + error_fraction * quant_error[k], 0, 1)

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
def dither_image(screen, float[:, :, ::1] image_rgb, dither, int lookahead, unsigned char verbose, float[:,::1] rgb_to_cam16ucs):
    cdef int y, x, i, j, k
    # cdef float[3] input_pixel_rgb
    cdef float[3] quant_error
    cdef unsigned char output_pixel_nbit
    cdef unsigned char best_next_pixels
    cdef float[3] output_pixel_rgb

    # Hoist some python attribute accesses into C variables for efficient access during the main loop

    cdef int yres = screen.Y_RES
    cdef int xres = screen.X_RES

    # TODO: convert this instead of storing on palette?
    cdef float[:, :, ::1] palette_cam16 = np.zeros((len(screen.palette.CAM16UCS), 4, 3), dtype=np.float32)
    for i, j in screen.palette.CAM16UCS.keys():
        for k in range(3):
            palette_cam16[i, j, k] = screen.palette.CAM16UCS[i, j][k]

    cdef float[:, :, ::1] palette_rgb = np.zeros((len(screen.palette.RGB), 4, 3), dtype=np.float32)
    for i, j in screen.palette.RGB.keys():
        for k in range(3):
            palette_rgb[i, j, k] = screen.palette.RGB[i, j][k] / 255

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
            #for i in range(3):
            #    input_pixel_rgb[i] = image_rgb[y,x,i]
            if lookahead:
                # Compute all possible 2**N choices of n-bit pixel colours for positions x .. x + lookahead
                # lookahead_palette_choices_nbit = lookahead_options(lookahead, output_pixel_nbit)
                # Apply error diffusion for each of these 2**N choices, and compute which produces the closest match
                # to the source image over the succeeding N pixels
                best_next_pixels = dither_lookahead(
                        &cdither, palette_cam16, palette_rgb, image_rgb, x, y, lookahead, output_pixel_nbit, xres, rgb_to_cam16ucs)
                # Apply best choice for next 1 pixel
                output_pixel_nbit = lookahead_pixels(output_pixel_nbit, best_next_pixels, lookahead=0)
            #else:
            #    # Choose the closest colour among the available n-bit palette options
            #    palette_choices_nbit = screen.pixel_palette_options(output_pixel_nbit, x)
            #    output_pixel_nbit = find_nearest_colour(input_pixel_rgb, palette_choices_nbit, distances)

            # Apply error diffusion from chosen output pixel value
            for i in range(3):
                output_pixel_rgb[i] = palette_rgb[output_pixel_nbit, x % 4, i]
                quant_error[i] = image_rgb[y,x,i] - output_pixel_rgb[i]
            apply(&cdither, xres, yres, x, y, image_rgb, quant_error)

            # Update image with our chosen image pixel
            image_nbit[y, x] = output_pixel_nbit
            for i in range(3):
                image_rgb[y, x, i] = output_pixel_rgb[i]

    free(cdither.pattern)
    return image_nbit, np.array(image_rgb)
