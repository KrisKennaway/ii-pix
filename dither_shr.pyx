# cython: infer_types=True
# cython: profile=False

cimport cython
import numpy as np

# TODO: move these into a common module
cdef float clip(float a, float min_value, float max_value) nogil:
    return min(max(a, min_value), max_value)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float[::1] convert_rgb_to_cam16ucs(float[:, ::1] rgb_to_cam16ucs, float r, float g, float b) nogil:
    cdef unsigned int rgb_24bit = (<unsigned int>(r*255) << 16) + (<unsigned int>(g*255) << 8) + <unsigned int>(b*255)
    return rgb_to_cam16ucs[rgb_24bit]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double colour_distance_squared(float[::1] colour1, float[::1] colour2) nogil:
    return (colour1[0] - colour2[0]) ** 2 + (colour1[1] - colour2[1]) ** 2 + (colour1[2] - colour2[2]) ** 2


@cython.boundscheck(False)
@cython.wraparound(False)
def dither_shr_perfect(
        float[:, :, ::1] input_rgb, float[:, ::1] full_palette_cam, float[:, ::1] full_palette_rgb,
        float[:,::1] rgb_to_cam16ucs):
    cdef int y, x, idx, best_colour_idx, i
    cdef double best_distance, distance, total_image_error
    cdef float[::1] best_colour_rgb, pixel_cam
    cdef float quant_error
    cdef float[:, ::1] palette_rgb, palette_cam

    cdef float[:, :, ::1] working_image = np.copy(input_rgb)
    cdef float[:, ::1] line_cam = np.zeros((320, 3), dtype=np.float32)

    cdef int palette_size = full_palette_rgb.shape[0]

    cdef float decay = 0.5
    cdef int floyd_steinberg = 1

    total_image_error = 0.0
    for y in range(200):
        for x in range(320):
            line_cam[x, :] = convert_rgb_to_cam16ucs(
                rgb_to_cam16ucs, working_image[y,x,0], working_image[y,x,1], working_image[y,x,2])

        for x in range(320):
            pixel_cam = convert_rgb_to_cam16ucs(
                rgb_to_cam16ucs, working_image[y, x, 0], working_image[y, x, 1], working_image[y, x, 2])

            best_distance = 1e9
            best_colour_idx = -1
            for idx in range(palette_size):
                distance = colour_distance_squared(pixel_cam, full_palette_cam[idx, :])
                if distance < best_distance:
                    best_distance = distance
                    best_colour_idx = idx
            best_colour_rgb = full_palette_rgb[best_colour_idx, :]
            total_image_error += best_distance

            for i in range(3):
                quant_error = working_image[y, x, i] - best_colour_rgb[i]

                working_image[y, x, i] = best_colour_rgb[i]
                if floyd_steinberg:
                    # Floyd-Steinberg dither
                    # 0 * 7
                    # 3 5 1
                    if x < 319:
                        working_image[y, x + 1, i] = clip(
                            working_image[y, x + 1, i] + quant_error * (7 / 16), 0, 1)
                    if y < 199:
                        if x > 0:
                            working_image[y + 1, x - 1, i] = clip(
                                working_image[y + 1, x - 1, i] + decay * quant_error * (3 / 16), 0, 1)
                        working_image[y + 1, x, i] = clip(
                            working_image[y + 1, x, i] + decay * quant_error * (5 / 16), 0, 1)
                        if x < 319:
                            working_image[y + 1, x + 1, i] = clip(
                                working_image[y + 1, x + 1, i] + decay * quant_error * (1 / 16), 0, 1)
                else:
                    # Jarvis
                    # 0 0 X 7 5
                    # 3 5 7 5 3
                    # 1 3 5 3 1
                    if x < 319:
                        working_image[y, x + 1, i] = clip(
                            working_image[y, x + 1, i] + quant_error * (7 / 48), 0, 1)
                    if x < 318:
                        working_image[y, x + 2, i] = clip(
                            working_image[y, x + 2, i] + quant_error * (5 / 48), 0, 1)
                    if y < 199:
                        if x > 1:
                            working_image[y + 1, x - 2, i] = clip(
                                working_image[y + 1, x - 2, i] + decay * quant_error * (3 / 48), 0,
                                1)
                        if x > 0:
                            working_image[y + 1, x - 1, i] = clip(
                                working_image[y + 1, x - 1, i] + decay * quant_error * (5 / 48), 0,
                                1)
                        working_image[y + 1, x, i] = clip(
                            working_image[y + 1, x, i] + decay * quant_error * (7 / 48), 0, 1)
                        if x < 319:
                            working_image[y + 1, x + 1, i] = clip(
                                working_image[y + 1, x + 1, i] + decay * quant_error * (5 / 48),
                                0, 1)
                        if x < 318:
                            working_image[y + 1, x + 2, i] = clip(
                                working_image[y + 1, x + 2, i] + decay * quant_error * (3 / 48),
                                0, 1)
                    if y < 198:
                        if x > 1:
                            working_image[y + 2, x - 2, i] = clip(
                                working_image[y + 2, x - 2, i] + decay * decay * quant_error * (1 / 48), 0,
                                1)
                        if x > 0:
                            working_image[y + 2, x - 1, i] = clip(
                                working_image[y + 2, x - 1, i] + decay * decay * quant_error * (3 / 48), 0,
                                1)
                        working_image[y + 2, x, i] = clip(
                            working_image[y + 2, x, i] + decay * decay * quant_error * (5 / 48), 0, 1)
                        if x < 319:
                            working_image[y + 2, x + 1, i] = clip(
                                working_image[y + 2, x + 1, i] + decay * decay * quant_error * (3 / 48),
                                0, 1)
                        if x < 318:
                            working_image[y + 2, x + 2, i] = clip(
                                working_image[y + 2, x + 2, i] + decay * decay * quant_error * (1 / 48),
                                0, 1)

    return total_image_error, working_image


@cython.boundscheck(False)
@cython.wraparound(False)
def dither_shr(
        float[:, :, ::1] input_rgb, float[:, :, ::1] palettes_cam, float[:, :, ::1] palettes_rgb,
        float[:,::1] rgb_to_cam16ucs):
    cdef int y, x, idx, best_colour_idx, best_palette, i
    cdef double best_distance, distance, total_image_error
    cdef float[::1] best_colour_rgb, pixel_cam
    cdef float quant_error
    cdef float[:, ::1] palette_rgb, palette_cam

    cdef (unsigned char)[:, ::1] output_4bit = np.zeros((200, 320), dtype=np.uint8)
    cdef float[:, :, ::1] working_image = np.copy(input_rgb)
    cdef float[:, ::1] line_cam = np.zeros((320, 3), dtype=np.float32)

    cdef int[::1] line_to_palette = np.zeros(200, dtype=np.int32)
    cdef double[::1] palette_line_errors = np.zeros(200, dtype=np.float64)
    cdef PaletteSelection palette_line

    cdef float decay = 0.5
    cdef int floyd_steinberg = 1

    best_palette = -1
    total_image_error = 0.0
    for y in range(200):
        for x in range(320):
            line_cam[x, :] = convert_rgb_to_cam16ucs(
                rgb_to_cam16ucs, working_image[y,x,0], working_image[y,x,1], working_image[y,x,2])

        palette_line = best_palette_for_line(line_cam, palettes_cam, best_palette)
        best_palette = palette_line.palette_idx
        palette_line_errors[y] = palette_line.total_error

        palette_rgb = palettes_rgb[best_palette, :, :]
        palette_cam = palettes_cam[best_palette, :, :]
        line_to_palette[y] = best_palette

        for x in range(320):
            pixel_cam = convert_rgb_to_cam16ucs(
                rgb_to_cam16ucs, working_image[y, x, 0], working_image[y, x, 1], working_image[y, x, 2])

            best_distance = 1e9
            best_colour_idx = -1
            for idx in range(16):
                distance = colour_distance_squared(pixel_cam, palette_cam[idx, :])
                if distance < best_distance:
                    best_distance = distance
                    best_colour_idx = idx
            best_colour_rgb = palette_rgb[best_colour_idx]
            output_4bit[y, x] = best_colour_idx
            total_image_error += best_distance

            for i in range(3):
                quant_error = working_image[y, x, i] - best_colour_rgb[i]

                working_image[y, x, i] = best_colour_rgb[i]
                if floyd_steinberg:
                    # Floyd-Steinberg dither
                    # 0 * 7
                    # 3 5 1
                    if x < 319:
                        working_image[y, x + 1, i] = clip(
                            working_image[y, x + 1, i] + quant_error * (7 / 16), 0, 1)
                    if y < 199:
                        if x > 0:
                            working_image[y + 1, x - 1, i] = clip(
                                working_image[y + 1, x - 1, i] + decay * quant_error * (3 / 16), 0, 1)
                        working_image[y + 1, x, i] = clip(
                            working_image[y + 1, x, i] + decay * quant_error * (5 / 16), 0, 1)
                        if x < 319:
                            working_image[y + 1, x + 1, i] = clip(
                                working_image[y + 1, x + 1, i] + decay * quant_error * (1 / 16), 0, 1)
                else:
                    # Jarvis
                    # 0 0 X 7 5
                    # 3 5 7 5 3
                    # 1 3 5 3 1
                    if x < 319:
                        working_image[y, x + 1, i] = clip(
                            working_image[y, x + 1, i] + quant_error * (7 / 48), 0, 1)
                    if x < 318:
                        working_image[y, x + 2, i] = clip(
                            working_image[y, x + 2, i] + quant_error * (5 / 48), 0, 1)
                    if y < 199:
                        if x > 1:
                            working_image[y + 1, x - 2, i] = clip(
                                working_image[y + 1, x - 2, i] + decay * quant_error * (3 / 48), 0,
                                1)
                        if x > 0:
                            working_image[y + 1, x - 1, i] = clip(
                                working_image[y + 1, x - 1, i] + decay * quant_error * (5 / 48), 0,
                                1)
                        working_image[y + 1, x, i] = clip(
                            working_image[y + 1, x, i] + decay * quant_error * (7 / 48), 0, 1)
                        if x < 319:
                            working_image[y + 1, x + 1, i] = clip(
                                working_image[y + 1, x + 1, i] + decay * quant_error * (5 / 48),
                                0, 1)
                        if x < 318:
                            working_image[y + 1, x + 2, i] = clip(
                                working_image[y + 1, x + 2, i] + decay * quant_error * (3 / 48),
                                0, 1)
                    if y < 198:
                        if x > 1:
                            working_image[y + 2, x - 2, i] = clip(
                                working_image[y + 2, x - 2, i] + decay * decay * quant_error * (1 / 48), 0,
                                1)
                        if x > 0:
                            working_image[y + 2, x - 1, i] = clip(
                                working_image[y + 2, x - 1, i] + decay * decay * quant_error * (3 / 48), 0,
                                1)
                        working_image[y + 2, x, i] = clip(
                            working_image[y + 2, x, i] + decay * decay * quant_error * (5 / 48), 0, 1)
                        if x < 319:
                            working_image[y + 2, x + 1, i] = clip(
                                working_image[y + 2, x + 1, i] + decay * decay * quant_error * (3 / 48),
                                0, 1)
                        if x < 318:
                            working_image[y + 2, x + 2, i] = clip(
                                working_image[y + 2, x + 2, i] + decay * decay * quant_error * (1 / 48),
                                0, 1)

    return np.array(output_4bit, dtype=np.uint8), line_to_palette, total_image_error, np.array(palette_line_errors, dtype=np.float64)


cdef struct PaletteSelection:
    int palette_idx
    double total_error


@cython.boundscheck(False)
@cython.wraparound(False)
cdef PaletteSelection best_palette_for_line(float [:, ::1] line_cam, float[:, :, ::1] palettes_cam, int last_palette_idx) nogil:
    cdef int palette_idx, best_palette_idx, palette_entry_idx, pixel_idx
    cdef double best_total_dist, total_dist, best_pixel_dist, pixel_dist
    cdef float[:, ::1] palette_cam
    cdef float[::1] pixel_cam

    best_total_dist = 1e9
    best_palette_idx = -1
    cdef int line_size = line_cam.shape[0]
    for palette_idx in range(16):
        palette_cam = palettes_cam[palette_idx, :, :]
        total_dist = 0
        for pixel_idx in range(line_size):
            pixel_cam = line_cam[pixel_idx]
            best_pixel_dist = 1e9
            for palette_entry_idx in range(16):
                pixel_dist = colour_distance_squared(pixel_cam, palette_cam[palette_entry_idx, :])
                if pixel_dist < best_pixel_dist:
                    best_pixel_dist = pixel_dist
            total_dist += best_pixel_dist
        if total_dist < best_total_dist:
            best_total_dist = total_dist
            best_palette_idx = palette_idx

    cdef PaletteSelection res
    res.palette_idx = best_palette_idx
    res.total_error = best_total_dist
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[::1] _convert_rgb12_iigs_to_cam(float [:, ::1] rgb12_iigs_to_cam16ucs, (unsigned char)[::1] point_rgb12) nogil:
    cdef int rgb12 = (point_rgb12[0] << 8) | (point_rgb12[1] << 4) | point_rgb12[2]
    return rgb12_iigs_to_cam16ucs[rgb12]


def convert_rgb12_iigs_to_cam(float [:, ::1] rgb12_iigs_to_cam16ucs, (unsigned char)[::1] point_rgb12) -> float[::1]:
    return _convert_rgb12_iigs_to_cam(rgb12_iigs_to_cam16ucs, point_rgb12)

import colour

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:, ::1] linear_to_srgb_array(float[:, ::1] a, float gamma=2.4):
    cdef int i, j
    cdef float[:, ::1] res = np.empty_like(a, dtype=np.float32)
    for i in range(res.shape[0]):
        for j in range(3):
            if a[i, j] <= 0.0031308:
                res[i, j] = a[i, j] * 12.92
            else:
                res[i, j] = 1.055 * a[i, j] ** (1.0 / gamma) - 0.055
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef (unsigned char)[:, ::1] _convert_cam16ucs_to_rgb12_iigs(float[:, ::1] point_cam):
    cdef float[:, ::1] rgb
    cdef (float)[:, ::1] rgb12_iigs

    # Convert CAM16UCS input to RGB.  Even though this dynamically constructs a path on the graph of colour conversions
    # every time, in practise this seems to have a negligible overhead compared to the actual conversion functions.
    with colour.utilities.suppress_warnings(python_warnings=True):
        rgb = colour.convert(point_cam, "CAM16UCS", "RGB").astype(np.float32)

    # TODO: precompute this conversion matrix since it's static.  This accounts for about 10% of the CPU time here.
    rgb12_iigs = np.clip(
        # Convert to Rec.601 R'G'B'
        colour.YCbCr_to_RGB(
            # Gamma correct and convert Rec.709 R'G'B' to YCbCr
            colour.RGB_to_YCbCr(
                linear_to_srgb_array(rgb), K=colour.WEIGHTS_YCBCR['ITU-R BT.709']),
            K=colour.WEIGHTS_YCBCR['ITU-R BT.601']), 0, 1).astype(np.float32) * 15
    return np.round(rgb12_iigs).astype(np.uint8)


def convert_cam16ucs_to_rgb12_iigs(float[:, ::1] point_cam):
    return _convert_cam16ucs_to_rgb12_iigs(point_cam)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def k_means_with_fixed_centroids(
    int n_clusters, int n_fixed, float[:, ::1] samples, (unsigned char)[:, ::1] initial_centroids, int max_iterations,
    float [:, ::1] rgb12_iigs_to_cam16ucs):

    cdef double error, best_error, total_error, last_total_error
    cdef int centroid_idx, closest_centroid_idx, i, point_idx

    cdef (unsigned char)[:, ::1] centroids_rgb12 = np.copy(initial_centroids)
    cdef (unsigned char)[:, ::1] new_centroids_rgb12

    cdef float[::1] point_cam
    cdef float[:, ::1] new_centroids_cam = np.empty((n_clusters - n_fixed, 3), dtype=np.float32)
    cdef float[:, ::1] centroid_cam_sample_positions_total
    cdef int[::1] centroid_sample_counts

    last_total_error = 1e9
    for iteration in range(max_iterations):
        total_error = 0.0
        centroid_cam_sample_positions_total = np.zeros((16, 3), dtype=np.float32)
        centroid_sample_counts = np.zeros(16, dtype=np.int32)

        # For each sample, associate it to the closest centroid.  We want to compute the mean of all associated samples
        # but we do this by accumulating the (coordinate vector) total and number of associated samples.
        #
        # Centroid positions are tracked in 4-bit //gs RGB colour space with distances measured in CAM16UCS colour
        # space.
        for point_idx in range(samples.shape[0]):
            point_cam = samples[point_idx, :]
            best_error = 1e9
            closest_centroid_idx = 0
            for centroid_idx in range(n_clusters):
                error = colour_distance_squared(
                    _convert_rgb12_iigs_to_cam(rgb12_iigs_to_cam16ucs, centroids_rgb12[centroid_idx, :]), point_cam)
                if error < best_error:
                    best_error = error
                    closest_centroid_idx = centroid_idx
            for i in range(3):
                centroid_cam_sample_positions_total[closest_centroid_idx, i] += point_cam[i]
            centroid_sample_counts[closest_centroid_idx] += 1
            total_error += best_error

        # Since the allowed centroid positions are discrete (and not uniformly spaced in CAM16UCS colour space), we
        # can't rely on measuring total centroid movement as a termination condition.  e.g. sometimes the nearest
        # available point to an intended next centroid position will increase the total distance, or centroids may
        # oscillate between two neighbouring positions.  Instead, we terminate when the total error stops decreasing.
        if total_error >= last_total_error:
            break
        last_total_error = total_error

        # Compute new centroid positions in CAM16UCS colour space
        for centroid_idx in range(n_fixed, n_clusters):
            if centroid_sample_counts[centroid_idx]:
                for i in range(3):
                    new_centroids_cam[centroid_idx - n_fixed, i] = (
                        centroid_cam_sample_positions_total[centroid_idx, i] / centroid_sample_counts[centroid_idx])

        # Convert all new centroids back to //gb RGB colour space (done as a single matrix since
        # _convert_cam16ucs_to_rgb12_iigs has nontrivial overhead)
        new_centroids_rgb12 = _convert_cam16ucs_to_rgb12_iigs(new_centroids_cam)

        # Update positions for non-fixed centroids
        for centroid_idx in range(n_clusters - n_fixed):
            for i in range(3):
                if centroids_rgb12[centroid_idx + n_fixed, i] != new_centroids_rgb12[centroid_idx, i]:
                    centroids_rgb12[centroid_idx + n_fixed, i] = new_centroids_rgb12[centroid_idx, i]

    return centroids_rgb12