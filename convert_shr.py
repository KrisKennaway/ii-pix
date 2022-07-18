from collections import defaultdict
import os.path
import random
from typing import Tuple

from PIL import Image
import colour
import numpy as np
from sklearn import cluster

from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

import dither_shr as dither_shr_pyx
import image as image_py


class ClusterPalette:
    def __init__(
            self, image: np.ndarray, rgb12_iigs_to_cam16ucs, rgb24_to_cam16ucs,
            fixed_colours=0):

        # Conversion matrix from 12-bit //gs RGB colour space to CAM16UCS
        # colour space
        self._rgb12_iigs_to_cam16ucs = rgb12_iigs_to_cam16ucs

        # Conversion matrix from 24-bit linear RGB colour space to CAM16UCS
        # colour space
        self._rgb24_to_cam16ucs = rgb24_to_cam16ucs

        # Preprocessed source image in 24-bit linear RGB colour space.  We
        # first dither the source image using the full 12-bit //gs RGB colour
        # palette, ignoring SHR palette limitations (i.e. 4096 independent
        # colours for each pixel).  This gives much better results for e.g.
        # solid blocks of colour, which would be dithered inconsistently if
        # targeting the source image directly.
        self._image_rgb = self._perfect_dither(image)

        # Preprocessed source image in CAM16UCS colour space
        self._colours_cam = self._image_colours_cam(self._image_rgb)

        # We fit a 16-colour palette against the entire image which is used
        # as starting values for fitting the reserved colours in the 16 SHR
        # palettes.
        self._global_palette = np.empty((16, 3), dtype=np.uint8)

        # How many image colours to fix identically across all 16 SHR
        # palettes.  These are taken to be the most prevalent colours from
        # _global_palette.
        self._fixed_colours = fixed_colours

        # 16 SHR palettes each of 16 colours, in CAM16UCS colour space
        self._palettes_cam = np.empty((16, 16, 3), dtype=np.float32)

        # 16 SHR palettes each of 16 colours, in //gs 4-bit RGB colour space
        self._palettes_rgb = np.empty((16, 16, 3), dtype=np.uint8)

        # defaultdict(list) mapping palette index to the lines that use this
        # palette
        self._palette_lines = self._init_palette_lines()

    @staticmethod
    def _image_colours_cam(image: Image):
        colours_rgb = np.asarray(image)  # .reshape((-1, 3))
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            colours_cam = colour.convert(colours_rgb, "RGB",
                                         "CAM16UCS").astype(np.float32)
        return colours_cam

    def _init_palette_lines(self, init_random=False):
        palette_lines = defaultdict(list)

        if init_random:
            lines = list(range(200))
            random.shuffle(lines)
            idx = 0
            while lines:
                palette_lines[idx].append(lines.pop())
                idx += 1
        else:
            palette_splits = self._equal_palette_splits()
            for i, lh in enumerate(palette_splits):
                l, h = lh
                palette_lines[i].extend(list(range(l, h)))
        return palette_lines

    @staticmethod
    def _equal_palette_splits(palette_height=35):
        # The 16 palettes are striped across consecutive (overlapping) line
        # ranges.  Since nearby lines tend to have similar colours, this has
        # the effect of smoothing out the colour transitions across palettes.

        # If we want to overlap 16 palettes in 200 lines, where each palette
        # has height H and overlaps the previous one by L lines, then the
        # boundaries are at lines:
        #   (0, H), (H-L, 2H-L), (2H-2L, 3H-2L), ..., (15H-15L, 16H - 15L)
        # i.e. 16H - 15L = 200, so for a given palette height H we need to
        # overlap by:
        #   L = (16H - 200)/15

        palette_overlap = (16 * palette_height - 200) / 15

        palette_ranges = []
        for palette_idx in range(16):
            palette_lower = palette_idx * (palette_height - palette_overlap)
            palette_upper = palette_lower + palette_height
            palette_ranges.append((int(np.round(palette_lower)),
                                   int(np.round(palette_upper))))
        return palette_ranges

    def _perfect_dither(self, source_image: np.ndarray):
        """Dither a "perfect" image using the full 12-bit //gs RGB colour
        palette, ignoring restrictions."""

        # Suppress divide by zero warning,
        # https://github.com/colour-science/colour/issues/900
        with colour.utilities.suppress_warnings(python_warnings=True):
            full_palette_linear_rgb = colour.convert(
                self._rgb12_iigs_to_cam16ucs, "CAM16UCS", "RGB").astype(
                np.float32)

        total_image_error, image_rgb = dither_shr_pyx.dither_shr_perfect(
            source_image, self._rgb12_iigs_to_cam16ucs, full_palette_linear_rgb,
            self._rgb24_to_cam16ucs)
        # print("Perfect image error:", total_image_error)
        return image_rgb

    def _dither_image(self, palettes_cam):
        # Suppress divide by zero warning,
        # https://github.com/colour-science/colour/issues/900
        with colour.utilities.suppress_warnings(python_warnings=True):
            palettes_linear_rgb = colour.convert(
                palettes_cam, "CAM16UCS", "RGB").astype(np.float32)

        output_4bit, line_to_palette, total_image_error, palette_line_errors = \
            dither_shr_pyx.dither_shr(
                self._image_rgb, palettes_cam, palettes_linear_rgb,
                self._rgb24_to_cam16ucs)

        # Update map of palettes to image lines for which the palette was the
        # best match
        palette_lines = defaultdict(list)
        for line, palette in enumerate(line_to_palette):
            palette_lines[palette].append(line)
        self._palette_lines = palette_lines

        self._palette_line_errors = palette_line_errors

        return (output_4bit, line_to_palette, palettes_linear_rgb,
                total_image_error)

    def iterate(self, max_inner_iterations: int,
                max_outer_iterations: int):
        total_image_error = 1e9

        outer_iterations_since_improvement = 0
        while outer_iterations_since_improvement < max_outer_iterations:
            inner_iterations_since_improvement = 0
            self._palette_lines = self._init_palette_lines()

            while inner_iterations_since_improvement < max_inner_iterations:
                # print("Iterations %d" % inner_iterations_since_improvement)
                new_palettes_cam, new_palettes_rgb12_iigs = (
                    self._fit_shr_palettes())

                # Recompute image with proposed palettes and check whether it
                # has lower total image error than our previous best.
                (output_4bit, line_to_palette, palettes_linear_rgb,
                 new_total_image_error) = self._dither_image(new_palettes_cam)

                self._reassign_unused_palettes(
                    line_to_palette, new_palettes_rgb12_iigs)

                if new_total_image_error >= total_image_error:
                    inner_iterations_since_improvement += 1
                    continue

                # We found a globally better set of palettes, so restart the
                # clocks
                inner_iterations_since_improvement = 0
                outer_iterations_since_improvement = -1
                total_image_error = new_total_image_error

                self._palettes_cam = new_palettes_cam
                self._palettes_rgb = new_palettes_rgb12_iigs

                yield (new_total_image_error, output_4bit, line_to_palette,
                       new_palettes_rgb12_iigs, palettes_linear_rgb)
            outer_iterations_since_improvement += 1

    def _fit_shr_palettes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Attempt to find new palettes that locally improve image quality.

        Re-fit a set of 16 palettes from (overlapping) line ranges of the
        source image, using k-means clustering in CAM16-UCS colour space.

        We maintain the total image error for the pixels on which the 16
        palettes are clustered.  A new palette that increases this local
        image error is rejected.

        New palettes that reduce local error cannot be applied immediately
        though, because they may cause an increase in *global* image error
        when dithering.  i.e. they would reduce the overall image quality.

        The current (locally) best palettes are returned and can be applied
        using accept_palettes()

        XXX update
        """
        new_palettes_cam = np.empty_like(self._palettes_cam)
        new_palettes_rgb12_iigs = np.empty_like(self._palettes_rgb)

        # Compute a new 16-colour global palette for the entire image,
        # used as the starting center positions for k-means clustering of the
        # individual palettes
        self._fit_global_palette()

        for palette_idx in range(16):
            palette_pixels = (
                self._colours_cam[self._palette_lines[
                                      palette_idx], :, :].reshape(-1, 3))

            # Fix reserved colours from the global palette.
            initial_centroids = np.copy(self._global_palette)
            pixels_rgb_iigs = dither_shr_pyx.convert_cam16ucs_to_rgb12_iigs(
                palette_pixels)
            seen_colours = set()
            for i in range(self._fixed_colours):
                seen_colours.add(tuple(initial_centroids[i, :]))

            # Pick unique random colours from the sample points for the
            # remaining initial centroids.
            for i in range(self._fixed_colours, 16):
                choice = np.random.randint(0, pixels_rgb_iigs.shape[0])
                new_colour = pixels_rgb_iigs[choice, :]
                if tuple(new_colour) in seen_colours:
                    continue
                seen_colours.add(tuple(new_colour))
                initial_centroids[i, :] = new_colour

            # If there are any single colours in our source //gs RGB pixels that
            # represent more than fixed_colour_fraction_threshold of the total,
            # then fix these colours for the palette instead of clustering
            # them.  This reduces artifacting on blocks of colour.
            fixed_colour_fraction_threshold = 0.1
            most_frequent_colours = sorted(list(zip(
                *np.unique(pixels_rgb_iigs, return_counts=True, axis=0))),
                key=lambda kv: kv[1], reverse=True)
            fixed_colours = self._fixed_colours
            for palette_colour, freq in most_frequent_colours:
                if (freq < (palette_pixels.shape[0] *
                            fixed_colour_fraction_threshold)) or (
                        fixed_colours == 16):
                    break
                if tuple(palette_colour) not in seen_colours:
                    seen_colours.add(tuple(palette_colour))
                    initial_centroids[fixed_colours, :] = palette_colour
                    fixed_colours += 1

            palette_rgb12_iigs = dither_shr_pyx.k_means_with_fixed_centroids(
                n_clusters=16, n_fixed=fixed_colours,
                samples=palette_pixels,
                initial_centroids=initial_centroids,
                max_iterations=1000,
                rgb12_iigs_to_cam16ucs=self._rgb12_iigs_to_cam16ucs)
            # If the k-means clustering returned fewer than 16 unique colours,
            # fill out the remainder with the most common pixels colours that
            # have not yet been used.
            palette_rgb12_iigs = self._fill_short_palette(
                palette_rgb12_iigs, most_frequent_colours)

            for i in range(16):
                new_palettes_cam[palette_idx, i, :] = (
                    np.array(dither_shr_pyx.convert_rgb12_iigs_to_cam(
                        self._rgb12_iigs_to_cam16ucs, palette_rgb12_iigs[
                            i]), dtype=np.float32))

            new_palettes_rgb12_iigs[palette_idx, :, :] = palette_rgb12_iigs

        self._palettes_accepted = False
        return new_palettes_cam, new_palettes_rgb12_iigs

    def _fit_global_palette(self):
        """Compute a 16-colour palette for the entire image to use as
        starting point for the sub-palettes.  This should help when the image
        has large blocks of colour since the sub-palettes will tend to pick the
        same colours."""

        clusters = cluster.MiniBatchKMeans(n_clusters=16, max_iter=10000)
        clusters.fit_predict(self._colours_cam.reshape(-1, 3))

        # Dict of {palette idx : frequency count}
        palette_freq = {idx: 0 for idx in range(16)}
        for idx, freq in zip(*np.unique(clusters.labels_, return_counts=True)):
            palette_freq[idx] = freq

        frequency_order = [
            k for k, v in sorted(
                list(palette_freq.items()), key=lambda kv: kv[1], reverse=True)]

        self._global_palette = (
            dither_shr_pyx.convert_cam16ucs_to_rgb12_iigs(
                clusters.cluster_centers_[frequency_order].astype(
                    np.float32)))

    @staticmethod
    def _fill_short_palette(palette_iigs_rgb, most_frequent_colours):
        """Fill out the palette to 16 unique entries."""

        # We want to maintain order of insertion so that we respect the
        # ordering of fixed colours in the palette.  Python doesn't have an
        # orderedset but dicts preserve insertion order.
        palette_set = {}
        for palette_entry in palette_iigs_rgb:
            palette_set[tuple(palette_entry)] = True
        if len(palette_set) == 16:
            return palette_iigs_rgb

        # Add most frequent image colours that are not yet in the palette
        for palette_colour, freq in most_frequent_colours:
            if tuple(palette_colour) in palette_set:
                continue
            palette_set[tuple(palette_colour)] = True
            if len(palette_set) == 16:
                break

        # We couldn't find any more unique colours, fill out with random ones.
        while len(palette_set) < 16:
            palette_set[
                tuple(np.random.randint(0, 16, size=3, dtype=np.uint8))] = True

        return np.array(tuple(palette_set.keys()), dtype=np.uint8)

    def _reassign_unused_palettes(self, line_to_palette, palettes_iigs_rgb):
        palettes_used = [False] * 16
        for palette in line_to_palette:
            palettes_used[palette] = True
        best_palette_lines = [v for k, v in sorted(list(zip(
            self._palette_line_errors, range(200))))]

        all_palettes = set()
        for palette_idx, palette_iigs_rgb in enumerate(palettes_iigs_rgb):
            palette_set = set()
            for palette_entry in palette_iigs_rgb:
                palette_set.add(tuple(palette_entry))
            palette_set = frozenset(palette_set)
            if palette_set in all_palettes:
                print("Duplicate palette", palette_idx, palette_set)
                palettes_used[palette_idx] = False

        for palette_idx, palette_used in enumerate(palettes_used):
            if palette_used:
                continue

            # TODO: also remove from old entry
            worst_line = best_palette_lines.pop()
            self._palette_lines[palette_idx] = [worst_line]


def convert(screen, rgb: np.ndarray, args):
    # Conversion matrix from RGB to CAM16UCS colour values.  Indexed by
    # 24-bit RGB value
    base_dir = os.path.dirname(__file__)
    rgb24_to_cam16ucs = np.load(
        os.path.join(base_dir, "data/rgb24_to_cam16ucs.npy"))
    rgb12_iigs_to_cam16ucs = np.load(
        os.path.join(base_dir, "data/rgb12_iigs_to_cam16ucs.npy"))

    # TODO: flags
    inner_iterations = 10
    outer_iterations = 20

    if args.show_output:
        pygame.init()
        canvas = pygame.display.set_mode((640, 400))
        canvas.fill((0, 0, 0))
        pygame.display.set_caption("][-Pix image preview")
        pygame.event.pump()  # Update caption
        pygame.display.flip()

    total_image_error = None
    cluster_palette = ClusterPalette(
        rgb, fixed_colours=args.fixed_colours,
        rgb12_iigs_to_cam16ucs=rgb12_iigs_to_cam16ucs,
        rgb24_to_cam16ucs=rgb24_to_cam16ucs)

    output_base, output_ext = os.path.splitext(args.output)

    seq = 0
    for (
            new_total_image_error, output_4bit, line_to_palette,
            palettes_rgb12_iigs,
            palettes_linear_rgb
    ) in cluster_palette.iterate(inner_iterations, outer_iterations):

        if args.verbose and total_image_error is not None:
            print("Improved quality +%f%% (%f)" % (
                (1 - new_total_image_error / total_image_error) * 100,
                new_total_image_error))
        total_image_error = new_total_image_error
        for i in range(16):
            screen.set_palette(i, palettes_rgb12_iigs[i, :, :])

        # Recompute current screen RGB image
        screen.set_pixels(output_4bit)
        output_rgb = np.empty((200, 320, 3), dtype=np.uint8)
        for i in range(200):
            screen.line_palette[i] = line_to_palette[i]
            output_rgb[i, :, :] = (
                    palettes_linear_rgb[line_to_palette[i]][
                        output_4bit[i, :]] * 255
            ).astype(np.uint8)

        output_srgb = (image_py.linear_to_srgb(output_rgb)).astype(np.uint8)
        out_image = image_py.resize(
            Image.fromarray(output_srgb), screen.X_RES * 2, screen.Y_RES * 2,
            srgb_output=True)

        if args.show_output:
            surface = pygame.surfarray.make_surface(
                np.asarray(out_image).transpose((1, 0, 2)))  # flip y/x axes
            canvas.blit(surface, (0, 0))
            pygame.display.set_caption("][-Pix image preview [Iteration %d]"
                                       % seq)
            pygame.event.pump()  # Update caption
            pygame.display.flip()

        unique_colours = np.unique(
            palettes_rgb12_iigs.reshape(-1, 3), axis=0).shape[0]
        if args.verbose:
            print("%d unique colours" % unique_colours)

        if args.save_preview:
            # Save super hi-res image
            if args.save_intermediate:
                outfile = "%s-%d-preview.png" % (output_base, seq)
            else:
                outfile = "%s-preview.png" % output_base
            out_image.save(outfile, "PNG")
        screen.pack()

        if args.save_intermediate:
            outfile = "%s-%d%s" % (output_base, seq, output_ext)
        else:
            outfile = "%s%s" % (output_base, output_ext)
        with open(outfile, "wb") as f:
            f.write(bytes(screen.memory))

        seq += 1

    if args.show_final_score:
        print("FINAL_SCORE:", total_image_error)
