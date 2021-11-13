# cython: infer_types=True
# cython: profile=False

cimport cython
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
        int x_res, float[:,::1] rgb_to_cam16ucs, unsigned char palette_depth) nogil:
    cdef int candidate_pixels, i, j
    cdef float[3] quant_error
    cdef int best
    cdef float best_error = 2**31-1
    cdef float total_error
    cdef unsigned char next_pixels
    cdef int phase
    cdef float[::1] lah_cam16ucs

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
    for candidate_pixels in range(1 << lookahead):
        # Working copy of input pixels
        for i in range(xxr - x):
            for j in range(3):
                lah_image_rgb[i * lah_shape2 + j] = image_rgb[y, x+i, j]

        total_error = 0
        # Apply dithering to lookahead horizon or edge of screen
        for i in range(xxr - x):
            xl = dither_bounds_xl(dither, i)
            xr = dither_bounds_xr(dither, xxr - x, i)
            phase = (x + i) % 4

            next_pixels = shift_pixel_window(
                    last_pixels, next_pixels=candidate_pixels, shift_right_by=i+1, window_width=palette_depth)

            # We don't update the input at position x (since we've already chosen fixed outputs), but we do propagate
            # quantization errors to positions >x  so we can compensate for how good/bad these choices were.  i.e. the
            # next_pixels choices are fixed, but we can still distribute quantization error from having made these
            # choices, in order to compute the total error.
            for j in range(3):
                quant_error[j] = lah_image_rgb[i * lah_shape2 + j] - palette_rgb[next_pixels, phase, j]
            apply_one_line(dither, xl, xr, i, lah_image_rgb, lah_shape2, quant_error)

            lah_cam16ucs = convert_rgb_to_cam16ucs(
                rgb_to_cam16ucs, lah_image_rgb[i*lah_shape2], lah_image_rgb[i*lah_shape2+1],
                lah_image_rgb[i*lah_shape2+2])
            total_error += colour_distance_squared(lah_cam16ucs, palette_cam16[next_pixels, phase])

            if total_error >= best_error:
                # No need to continue
                break

        if total_error < best_error:
            best_error = total_error
            best = candidate_pixels

    free(lah_image_rgb)
    return best


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float[::1] convert_rgb_to_cam16ucs(float[:, ::1] rgb_to_cam16ucs, float r, float g, float b) nogil:
    cdef int rgb_24bit = (<int>(r*255) << 16) + (<int>(g*255) << 8) + <int>(b*255)
    return rgb_to_cam16ucs[rgb_24bit]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float fabs(float value) nogil:
    return -value if value < 0 else value


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float colour_distance_squared(float[::1] colour1, float[::1] colour2) nogil:
    return (colour1[0] - colour2[0]) ** 2 + (colour1[1] - colour2[1]) ** 2 + (colour1[2] - colour2[2]) ** 2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float colour_distance(float[::1] colour1, float[::1] colour2) nogil:
    return fabs(colour1[0] - colour2[0]) + fabs(colour1[1] - colour2[1]) + fabs(colour1[2] - colour2[2])


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
    for i in range(yt, yb):
        for j in range(xl, xr):
            error_fraction = dither.pattern[(i - y) * dither.x_shape + j - x + dither.x_origin]
            for k in range(3):
                image[i,j,k] = clip(image[i,j,k] + error_fraction * quant_error[k], 0, 1)


@cython.boundscheck(False)
@cython.wraparound(False)
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
@cython.boundscheck(False)
@cython.wraparound(False)
def dither_image(
        screen, float[:, :, ::1] image_rgb, dither, int lookahead, unsigned char verbose, float[:,::1] rgb_to_cam16ucs):
    cdef int y, x
    cdef unsigned char i, j, pixels_nbit, phase
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

    for y in range(yres):
        if verbose:
            print("%d/%d" % (y, yres))
        output_pixel_nbit = 0
        for x in range(xres):
            # Compute all possible 2**N choices of n-bit pixel colours for positions x .. x + lookahead
            # lookahead_palette_choices_nbit = lookahead_options(lookahead, output_pixel_nbit)
            # Apply error diffusion for each of these 2**N choices, and compute which produces the closest match
            # to the source image over the succeeding N pixels
            best_next_pixels = dither_lookahead(
                    &cdither, palette_cam16, palette_rgb, image_rgb, x, y, lookahead, output_pixel_nbit, xres,
                    rgb_to_cam16ucs, palette_depth)
            # Apply best choice for next 1 pixel
            output_pixel_nbit = shift_pixel_window(
                    output_pixel_nbit, best_next_pixels, shift_right_by=1, window_width=palette_depth)

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
    return image_nbit_to_bitmap(image_nbit, xres, yres, palette_depth)

import colour

@cython.boundscheck(False)
@cython.wraparound(False)
def dither_shr(float[:, :, ::1] working_image, float[:, :, ::1] palettes_cam, float[:, :, ::1] palettes_rgb, float[:,::1] rgb_to_cam16ucs):
    cdef int y, x, idx, best_colour_idx, best_palette
    cdef float best_distance, distance
    cdef float[::1] best_colour_rgb, pixel_cam, colour_rgb, colour_cam
    cdef float quant_error
    cdef float[:, ::1] palette_rgb

    cdef (unsigned char)[:, ::1] output_4bit = np.zeros((200, 320), dtype=np.uint8)
    # cdef (unsigned char)[:, :, ::1] output_rgb = np.zeros((200, 320, 3), dtype=np.uint8)

    cdef float[:, ::1] line_cam = np.zeros((320, 3), dtype=np.float32)

    cdef int[::1] line_to_palette = np.zeros(200, dtype=np.int32)
    best_palette = 15
    for y in range(200):
        print(y)
        for x in range(320):
            colour_cam = convert_rgb_to_cam16ucs(
                rgb_to_cam16ucs, working_image[y,x,0], working_image[y,x,1], working_image[y,x,2])
            line_cam[x, :] = colour_cam

        best_palette = best_palette_for_line(line_cam, palettes_cam, <int>(y * 16 / 200), best_palette)
        # print("-->", best_palette)
        palette_rgb = palettes_rgb[best_palette, :, :]
        line_to_palette[y] = best_palette

        for x in range(320):
            pixel_cam = convert_rgb_to_cam16ucs(
                rgb_to_cam16ucs, working_image[y, x, 0], working_image[y, x, 1], working_image[y, x, 2])

            best_distance = 1e9
            best_colour_idx = -1
            for idx in range(16):
                colour_rgb = palette_rgb[idx, :]
                colour_cam = convert_rgb_to_cam16ucs(rgb_to_cam16ucs, colour_rgb[0], colour_rgb[1], colour_rgb[2])
                distance = colour_distance_squared(pixel_cam, colour_cam)
                if distance < best_distance:
                    best_distance = distance
                    best_colour_idx = idx
            best_colour_rgb = palette_rgb[best_colour_idx]
            output_4bit[y, x] = best_colour_idx

            for i in range(3):
                quant_error = working_image[y, x, i] - best_colour_rgb[i]

                # Floyd-Steinberg dither
                # 0 * 7
                # 3 5 1
                working_image[y, x, i] = best_colour_rgb[i]
                if x < 319:
                    working_image[y, x + 1, i] = clip(
                        working_image[y, x + 1, i] + quant_error * (7 / 16), 0, 1)
                if y < 199:
                    if x > 0:
                        working_image[y + 1, x - 1, i] = clip(
                            working_image[y + 1, x - 1, i] + quant_error * (3 / 32), 0, 1)
                    working_image[y + 1, x, i] = clip(
                        working_image[y + 1, x, i] + quant_error * (5 / 32), 0, 1)
                    if x < 319:
                        working_image[y + 1, x + 1, i] = clip(
                            working_image[y + 1, x + 1, i] + quant_error * (1 / 32), 0, 1)

#                # 0 0 X 7 5
#                # 3 5 7 5 3
#                # 1 3 5 3 1
                #if x < 319:
                #    working_image[y, x + 1, i] = clip(
                #        working_image[y, x + 1, i] + quant_error * (7 / 48), 0, 1)
                #if x < 318:
                #    working_image[y, x + 2, i] = clip(
                #        working_image[y, x + 2, i] + quant_error * (5 / 48), 0, 1)
                #if y < 199:
                #    if x > 1:
                #        working_image[y + 1, x - 2, i] = clip(
                #            working_image[y + 1, x - 2, i] + quant_error * (3 / 48), 0,
                #            1)
                #    if x > 0:
                #        working_image[y + 1, x - 1, i] = clip(
                #            working_image[y + 1, x - 1, i] + quant_error * (5 / 48), 0,
                #            1)
                #    working_image[y + 1, x, i] = clip(
                #        working_image[y + 1, x, i] + quant_error * (7 / 48), 0, 1)
                #    if x < 319:
                #        working_image[y + 1, x + 1, i] = clip(
                #            working_image[y + 1, x + 1, i] + quant_error * (5 / 48),
                #            0, 1)
                #    if x < 318:
                #        working_image[y + 1, x + 2, i] = clip(
                #            working_image[y + 1, x + 2, i] + quant_error * (3 / 48),
                #            0, 1)
                #if y < 198:
                #    if x > 1:
                #        working_image[y + 2, x - 2, i] = clip(
                #            working_image[y + 2, x - 2, i] + quant_error * (1 / 48), 0,
                #            1)
                #    if x > 0:
                #        working_image[y + 2, x - 1, i] = clip(
                #            working_image[y + 2, x - 1, i] + quant_error * (3 / 48), 0,
                #            1)
                #    working_image[y + 2, x, i] = clip(
                #        working_image[y + 2, x, i] + quant_error * (5 / 48), 0, 1)
                #    if x < 319:
                #        working_image[y + 2, x + 1, i] = clip(
                #            working_image[y + 2, x + 1, i] + quant_error * (3 / 48),
                #            0, 1)
                #    if x < 318:
                #        working_image[y + 2, x + 2, i] = clip(
                #            working_image[y + 2, x + 2, i] + quant_error * (1 / 48),
                #            0, 1)

    return np.array(output_4bit, dtype=np.uint8), line_to_palette

import collections
import random

@cython.boundscheck(False)
@cython.wraparound(False)
def k_means_with_fixed_centroids(
        int n_clusters, float[:, ::1] data, float[:, ::1] fixed_centroids = None,
        int iterations = 10000, float tolerance = 1e-3):
    cdef int i, iteration, centroid_idx, num_fixed_centroids, num_random_centroids, best_centroid_idx
    cdef float[::1] point, centroid, new_centroid, old_centroid
    cdef float[:, ::1] centroids
    cdef float best_dist, centroid_movement, dist

    centroids = np.zeros((n_clusters, 3), dtype=np.float32)
    if fixed_centroids is not None:
        centroids[:fixed_centroids.shape[0], :] = fixed_centroids
    num_fixed_centroids = fixed_centroids.shape[0] if fixed_centroids is not None else 0
    num_random_centroids = n_clusters - num_fixed_centroids

    # TODO: kmeans++ initialization
    cdef int rand_idx = random.randint(0, data.shape[0])
    for i in range(num_random_centroids):
        centroids[num_fixed_centroids + i, :] = data[rand_idx, :]

    cdef int[::1] centroid_weights = np.zeros(n_clusters, dtype=np.int32)
    for iteration in range(iterations):
        # print("centroids ", centroids)
        closest_points = collections.defaultdict(list)
        for point in data:
            best_dist = 1e9
            best_centroid_idx = 0
            for centroid_idx in range(n_clusters):
                centroid = centroids[centroid_idx, :]
                dist = colour_distance(centroid, point)
                if dist < best_dist:
                    best_dist = dist
                    best_centroid_idx = centroid_idx
            closest_points[best_centroid_idx].append(point)

        centroid_movement = 0
        for centroid_idx, points in closest_points.items():
            centroid_weights[centroid_idx] = len(points)
            if centroid_idx < num_fixed_centroids:
                continue
            new_centroid = np.median(np.array(points), axis=0)
            old_centroid = centroids[centroid_idx]
            centroid_movement += colour_distance(old_centroid, new_centroid)
            centroids[centroid_idx, :] = new_centroid
        # print("iteration %d: movement %f" % (iteration, centroid_movement))
        if centroid_movement < tolerance:
            break

    weighted_centroids = list(zip(centroid_weights, [tuple(c) for c in centroids]))
    print(weighted_centroids)
    return np.array([c for w, c in sorted(weighted_centroids, reverse=True)], dtype=np.float32)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int best_palette_for_line(float [:, ::1] line_cam, float[:, :, ::1] palettes_cam, int base_palette_idx, int last_palette_idx) nogil:
    cdef int palette_idx, best_palette_idx, palette_entry_idx, pixel_idx
    cdef float best_total_dist, total_dist, best_pixel_dist, pixel_dist
    cdef float[:, ::1] palette_cam
    cdef float[::1] pixel_cam, palette_entry

    best_total_dist = 1e9
    best_palette_idx = -1
    cdef int line_size = line_cam.shape[0]
    for palette_idx in range(16):
        palette_cam = palettes_cam[palette_idx, :, :]
        if palette_idx < (base_palette_idx - 1) or palette_idx > (base_palette_idx + 1):
            continue
        if palette_idx == last_palette_idx:
            continue
        total_dist = 0
        best_pixel_dist = 1e9
        for pixel_idx in range(line_size):
            pixel_cam = line_cam[pixel_idx]
            for palette_entry_idx in range(16):
                palette_entry = palette_cam[palette_entry_idx, :]
                pixel_dist = colour_distance_squared(pixel_cam, palette_entry)
                if pixel_dist < best_pixel_dist:
                    best_pixel_dist = pixel_dist
            total_dist += best_pixel_dist
            # print(total_dist)
        if total_dist < best_total_dist:
            best_total_dist = total_dist
            best_palette_idx = palette_idx
    return best_palette_idx

