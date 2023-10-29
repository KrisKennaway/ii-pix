# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
import numpy as np
from libc.stdlib cimport malloc, free

cimport common

import screen as screen_py


# TODO: use a cdef class
# C representation of dither_pattern.DitherPattern data, for efficient access.
cdef struct Dither:

    float* pattern   # Flattened dither pattern
    int x_shape
    int y_shape
    int x_origin
    int y_origin


# Compute left-hand bounding box for dithering at horizontal position x.
cdef inline int dither_bounds_xl(Dither *dither, int x) nogil:
    cdef int el = max(dither.x_origin - x, 0)
    cdef int xl = x - dither.x_origin + el
    return xl


#Compute right-hand bounding box for dithering at horizontal position x.
cdef inline int dither_bounds_xr(Dither *dither, int x_res, int x) nogil:
    cdef int er = min(dither.x_shape, x_res - x)
    cdef int xr = x - dither.x_origin + er
    return xr


# Compute upper bounding box for dithering at vertical position y.
cdef inline int dither_bounds_yt(Dither *dither, int y) nogil:
    cdef int et = max(dither.y_origin - y, 0)
    cdef int yt = y - dither.y_origin + et

    return yt


# Compute lower bounding box for dithering at vertical position y.
cdef inline int dither_bounds_yb(Dither *dither, int y_res, int y) nogil:
    cdef int eb = min(dither.y_shape, y_res - y)
    cdef int yb = y - dither.y_origin + eb
    return yb


cdef inline unsigned char shift_pixel_window(
        unsigned char last_pixels,
        unsigned int next_pixels,
        unsigned char shift_right_by,
        unsigned char window_width) nogil:
    """Right-shift a sliding window of n pixels to incorporate new pixels.

    Args:
        last_pixels: n-bit value representing n pixels from left up to current position (MSB = current pixel).
        next_pixels: n-bit value representing n pixels to right of current position (LSB = pixel to right)
        shift_right_by: how many pixels of next_pixels to shift into the sliding window
        window_width: how many pixels to maintain in the sliding window (must be <= 8)

    Returns: n-bit value representing shifted pixel window
    """
    cdef unsigned char window_mask = 0xff >> (8 - window_width)
    cdef unsigned int shifted_next_pixels

    if window_width > shift_right_by:
        shifted_next_pixels = next_pixels << (window_width - shift_right_by)
    else:
        shifted_next_pixels = next_pixels >> (shift_right_by - window_width)
    return ((last_pixels >> shift_right_by) | shifted_next_pixels) & window_mask


# Given a byte to store on the hi-res screen, compute the sequence of 560-resolution pixels that will be displayed.
# Hi-res graphics works like this:
# - Each of the low 7 bits in screen_byte results in enabling or disabling two sequential 560-resolution pixels.
# - pixel screen order is from LSB to MSB
# - if bit 8 (the "palette bit) is set then the 14-pixel sequence is shifted one position to the right, and the
#   left-most pixel is filled in by duplicating the right-most pixel controlled by the previous screen byte (i.e. bit 7)
# - this gives a 15 or 14 pixel sequence depending on whether or not the palette bit is set.
cdef unsigned int compute_fat_pixels(unsigned int screen_byte, unsigned char last_pixels) nogil:
    cdef int i, bit, fat_bit
    cdef unsigned int result = 0

    for i in range(7):
        bit = (screen_byte >> i) & 0b1
        fat_bit = bit << 1 | bit
        result |= (fat_bit) << (2 * i)
    if screen_byte & 0x80:
        # Palette bit shifts to the right
        result <<= 1
        result |= (last_pixels >> 7)

    return result


# Context parametrizes the differences between DHGR and HGR image optimization
cdef struct Context:
    # How many bit positions to lookahead when optimizing
    unsigned char bit_lookahead
    # How many screen pixels produced by bit_lookahead.  This is 1:1 for DHGR but for HGR 8 bits in memory produce
    # 14 or 15 screen pixels (see compute_fat_pixels above)
    unsigned char pixel_lookahead
    # HGR has a NTSC phase shift relative to DHGR which rotates the effective mappings from screen pixels to colours
    unsigned char phase_shift
    # Non-zero for HGR optimization
    unsigned char is_hgr


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
@cython.cdivision(True)
cdef int dither_lookahead(Dither* dither, unsigned char palette_depth, float[:, :, ::1] palette_cam16,
        float[:, :, ::1] palette_rgb, float[:, :, ::1] image_rgb, int x, int y, unsigned char last_pixels,
        int x_res, float[:,::1] rgb_to_cam16ucs, Context context) nogil:
    cdef int candidate, next_pixels, i, j
    cdef float[3] quant_error
    cdef int best
    cdef float best_error = 2**31-1
    cdef float total_error
    cdef unsigned char current_pixels
    cdef int phase
    cdef common.float3 lah_cam16ucs
    cdef float[3] cam

    # Don't bother dithering past the lookahead horizon or edge of screen.
    cdef int xxr = min(x + context.pixel_lookahead, x_res)

    cdef int lah_shape1 = xxr - x
    cdef int lah_shape2 = 3
    # TODO: try again with memoryview - does it actually have overhead here?
    cdef float *lah_image_rgb = <float *> malloc(lah_shape1 * lah_shape2 * sizeof(float))

    # For each 2**lookahead possibilities for the on/off state of the next lookahead pixels, apply error diffusion
    # and compute the total squared error to the source image.  Since we only have two possible colours for each
    # given pixel (dependent on the state already chosen for pixels to the left), we need to look beyond local minima.
    # i.e. it might be better to make a sub-optimal choice for this pixel if it allows access to much better pixel
    # colours at later positions.
    for candidate in range(1 << context.bit_lookahead):
        # Working copy of input pixels
        for i in range(xxr - x):
            for j in range(3):
                lah_image_rgb[i * lah_shape2 + j] = image_rgb[y, x+i, j]

        total_error = 0

        if context.is_hgr:
            # A HGR screen byte controls 14 or 15 screen pixels
            next_pixels = compute_fat_pixels(candidate, last_pixels)
        else:
            # DHGR pixels are 1:1 with memory bits
            next_pixels = candidate

        # Apply dithering to lookahead horizon or edge of screen
        for i in range(xxr - x):
            xl = dither_bounds_xl(dither, i)
            xr = dither_bounds_xr(dither, xxr - x, i)
            phase = (x + i + context.phase_shift) % 4

            current_pixels = shift_pixel_window(
                    last_pixels, next_pixels=next_pixels, shift_right_by=i+1, window_width=palette_depth)

            # We don't update the input at position x (since we've already chosen fixed outputs), but we do propagate
            # quantization errors to positions >x  so we can compensate for how good/bad these choices were.  i.e. the
            # current_pixels choices are fixed, but we can still distribute quantization error from having made these
            # choices, in order to compute the total error.
            for j in range(3):
                quant_error[j] = lah_image_rgb[i * lah_shape2 + j] - palette_rgb[current_pixels, phase, j]
            apply_one_line(dither, xl, xr, i, lah_image_rgb, lah_shape2, quant_error)

            # Accumulate error distance from pixel colour to target colour in CAM16UCS colour space
            lah_cam16ucs = common.convert_rgb_to_cam16ucs(
                rgb_to_cam16ucs, lah_image_rgb[i*lah_shape2], lah_image_rgb[i*lah_shape2+1],
                lah_image_rgb[i*lah_shape2+2])
            for j in range(3):
                cam[j] = palette_cam16[current_pixels, phase, j]
            total_error += common.colour_distance_squared(lah_cam16ucs.data, cam)

            if total_error >= best_error:
                # No need to continue
                break

        if total_error < best_error:
            best_error = total_error
            best = candidate

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
cdef inline void apply_one_line(Dither* dither, int xl, int xr, int x, float[] image, int image_shape1,
        float[] quant_error) noexcept nogil:

    cdef int i, j
    cdef float error_fraction

    for i in range(xl, xr):
        error_fraction = dither.pattern[i - x + dither.x_origin]
        for j in range(3):
            image[i * image_shape1 + j] = common.clip(image[i * image_shape1 + j] + error_fraction * quant_error[j], 0, 1)


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
cdef void apply(Dither* dither, int x_res, int y_res, int x, int y, float[:,:,::1] image, float[] quant_error) noexcept nogil:

    cdef int i, j, k

    cdef int yt = dither_bounds_yt(dither, y)
    cdef int yb = dither_bounds_yb(dither, y_res, y)
    cdef int xl = dither_bounds_xl(dither, x)
    cdef int xr = dither_bounds_xr(dither, x_res, x)

    cdef float error_fraction
    for i in range(yt, yb):
        for j in range(xl, xr):
            error_fraction = dither.pattern[(i - y) * dither.x_shape + j - x + dither.x_origin]
            for k in range(3):
                image[i,j,k] = common.clip(image[i,j,k] + error_fraction * quant_error[k], 0, 1)


cdef image_nbit_to_bitmap(
    (unsigned char)[:, ::1] image_nbit, unsigned int x_res, unsigned int y_res, unsigned char palette_depth):
    cdef unsigned int x, y
    bitmap = np.zeros((y_res, x_res), dtype=bool)
    for y in range(y_res):
        for x in range(x_res):
            # MSB of each array element is the pixel state at (x, y)
            bitmap[y, x] = image_nbit[y, x] >> (palette_depth - 1)
    return bitmap


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
@cython.cdivision(True)
def dither_image(
        screen, float[:, :, ::1] image_rgb, dither, int lookahead, unsigned char verbose, float[:, ::1] rgb_to_cam16ucs):
    cdef int y, x
    cdef unsigned char i, j, pixels_nbit, phase
    cdef float[3] quant_error
    cdef unsigned char output_pixel_nbit
    cdef unsigned int next_pixels
    cdef float[3] output_pixel_rgb

    # Hoist some python attribute accesses into C variables for efficient access during the main loop

    cdef int yres = screen.Y_RES
    cdef int xres = screen.X_RES

    # TODO: convert this instead of storing on palette?
    cdef float[:, :, ::1] palette_cam16 = np.zeros((len(screen.palette.CAM16UCS), 4, 3), dtype=np.float32)
    for pixels_nbit, phase in screen.palette.CAM16UCS.keys():
        for i in range(3):
            palette_cam16[pixels_nbit, phase, i] = screen.palette.CAM16UCS[pixels_nbit, phase][i]

    cdef float[:, :, ::1] palette_rgb = np.zeros((len(screen.palette.RGB), 4, 3), dtype=np.float32)
    for pixels_nbit, phase in screen.palette.RGB.keys():
        for i in range(3):
            palette_rgb[pixels_nbit, phase, i] = screen.palette.RGB[pixels_nbit, phase][i] / 255

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

    cdef unsigned char palette_depth = screen.palette.PALETTE_DEPTH

    # The nbit image representation contains the trailing n dot values as an n-bit value with MSB representing the
    # current pixel.  This choice (cf LSB) is slightly awkward but matches the DHGR behaviour that bit positions in
    # screen memory map LSB to MSB from L to R.  The value of n is chosen by the palette depth, i.e. how many trailing
    # dot positions are used to determine the colour of a given pixel.
    cdef (unsigned char)[:, ::1] image_nbit = np.empty((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

    cdef Context context
    if screen.MODE == screen_py.Mode.HI_RES:
        context.is_hgr = 1
        context.bit_lookahead = 8
        context.pixel_lookahead = 15
        # HGR and DHGR have a timing phase shift which rotates the effective mappings from screen dots to colours
        context.phase_shift = 3
    else:
        context.is_hgr = 0
        context.bit_lookahead = lookahead
        context.pixel_lookahead = lookahead
        context.phase_shift = 0

    cdef (unsigned char)[:, ::1] linear_bytemap = np.zeros((192, 40), dtype=np.uint8)

    # After performing lookahead, move ahead this many pixels at once.
    cdef int apply_batch_size
    if context.is_hgr:
        # For HGR we have to apply an entire screen byte at a time, which controls 14 or 15 pixels (see
        # compute_fat_pixels above).  This is because the high bit shifts this entire group of 14 pixels at once,
        # so we have to make a single decision about whether or not to enable it.
        apply_batch_size = 14
    else:
        # For DHGR we can choose each pixel state independently, so we get better results if we apply one pixel at
        # a time.
        apply_batch_size = 1

    for y in range(yres):
        if verbose:
            print("%d/%d" % (y, yres))
        output_pixel_nbit = 0
        for x in range(xres):
            if x % apply_batch_size == 0:
                # Compute all possible 2**N choices of n-bit pixel colours for positions x .. x + lookahead
                # Apply error diffusion for each of these 2**N choices, and compute which produces the closest match
                # to the source image over the succeeding N pixels
                next_pixels = dither_lookahead(
                        &cdither, palette_depth, palette_cam16, palette_rgb, image_rgb, x, y, output_pixel_nbit, xres,
                        rgb_to_cam16ucs, context)
                if context.is_hgr:
                    linear_bytemap[y, x // 14] = next_pixels
                    next_pixels = compute_fat_pixels(next_pixels, output_pixel_nbit)

            # Apply best choice for next 1 pixel
            output_pixel_nbit = shift_pixel_window(
                    output_pixel_nbit, next_pixels, shift_right_by=x % apply_batch_size + 1, window_width=palette_depth)
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
    return image_nbit_to_bitmap(image_nbit, xres, yres, palette_depth), linear_bytemap
