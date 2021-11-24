"""Image converter to Apple II Double Hi-Res format."""

import argparse
from collections import defaultdict
import os.path
import random
from typing import Tuple, List

from PIL import Image
import colour
import numpy as np
from sklearn import cluster

from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

import dither as dither_pyx
import dither_pattern
import image as image_py
import palette as palette_py
import screen as screen_py


# TODO:
# - support LR/DLR
# - support HGR


class ClusterPalette:
    def __init__(
            self, image: Image, rgb12_iigs_to_cam16ucs, rgb24_to_cam16ucs,
            reserved_colours=0):
        self._image_rgb = image
        self._colours_cam = self._image_colours_cam(image)

        self._errors = [1e9] * 16

        # We fit a 16-colour palette against the entire image which is used
        # as starting values for fitting the 16 SHR palettes.  This helps to
        # provide better global consistency of colours across the palettes,
        # e.g. for large blocks of colour.  Otherwise these can take a while
        # to converge.
        self._global_palette = np.empty((16, 3), dtype=np.uint8)

        # How many image colours to fix identically across all 16 SHR
        # palettes.  These are taken to be the most prevalent colours from
        # _global_palette.
        self._reserved_colours = reserved_colours

        # 16 SHR palettes each of 16 colours, in CAM16UCS format
        self._palettes_cam = np.empty((16, 16, 3), dtype=np.float32)

        # 16 SHR palettes each of 16 colours, in //gs 4-bit RGB format
        self._palettes_rgb = np.empty((16, 16, 3), dtype=np.uint8)

        # Conversion matrix from 12-bit //gs RGB colour space to CAM16UCS
        # colour space
        self._rgb12_iigs_to_cam16ucs = rgb12_iigs_to_cam16ucs

        self._rgb24_to_cam16ucs = rgb24_to_cam16ucs

        self._palette_lines = defaultdict(list)

        # List of line ranges used to train the 16 SHR palettes
        # [(lower_0, upper_0), ...]
        self._palette_splits = self._equal_palette_splits()
        self._init_palette_lines()

        # Whether the previous iteration of proposed palettes was accepted
        self._palettes_accepted = False

        # Which palette index's line ranges did we mutate in previous iteration
        self._palette_mutate_idx = 0

        # Delta applied to palette split in previous iteration
        self._palette_mutate_delta = (0, 0)

    def _init_palette_lines(self):
        for i, lh in enumerate(self._palette_splits):
            l, h = lh
            self._palette_lines[i].extend(list(range(l, h)))

    def _image_colours_cam(self, image: Image):
        colours_rgb = np.asarray(image)  # .reshape((-1, 3))
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            colours_cam = colour.convert(colours_rgb, "RGB",
                                         "CAM16UCS").astype(np.float32)
        return colours_cam

    def _equal_palette_splits(self, palette_height=35):
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

    def _dither_image(self, palettes_cam, penalty):
        # Suppress divide by zero warning,
        # https://github.com/colour-science/colour/issues/900
        with colour.utilities.suppress_warnings(python_warnings=True):
            palettes_linear_rgb = colour.convert(
                palettes_cam, "CAM16UCS", "RGB").astype(np.float32)

        output_4bit, line_to_palette, total_image_error, palette_line_errors = \
            dither_pyx.dither_shr(
                self._image_rgb, palettes_cam, palettes_linear_rgb,
                self._rgb24_to_cam16ucs, float(penalty))

        palette_lines = defaultdict(list)
        for line, palette in enumerate(line_to_palette):
            palette_lines[palette].append(line)
        self._palette_lines = palette_lines

        self._palette_line_errors = palette_line_errors

        return (output_4bit, line_to_palette, palettes_linear_rgb,
                total_image_error)

    def iterate(self, penalty: float, max_iterations: int):
        total_image_error = 1e9
        last_good_splits = self._palette_splits
        while True:
            print("New iteration")
            iterations_since_improvement = 0
            self._palette_splits = self._equal_palette_splits()
            self._init_palette_lines()

            self._fit_global_palette()
            while iterations_since_improvement < max_iterations:
                # print("Iterations %d" % iterations_since_improvement)
                new_palettes_cam, new_palettes_rgb12_iigs, new_palette_errors = (
                    self._propose_palettes())

                # Recompute image with proposed palettes and check whether it has
                # lower total image error than our previous best.
                (output_4bit, line_to_palette, palettes_linear_rgb,
                 new_total_image_error) = self._dither_image(
                    new_palettes_cam, penalty)

                self._reassign_unused_palettes(line_to_palette,
                                               last_good_splits)

                # print(total_image_error, new_total_image_error)
                if new_total_image_error >= total_image_error:
                    iterations_since_improvement += 1
                    continue

                # We found a globally better set of palettes
                iterations_since_improvement = 0
                last_good_splits = self._palette_splits
                total_image_error = new_total_image_error

                self._palettes_cam = new_palettes_cam
                self._palettes_rgb = new_palettes_rgb12_iigs
                self._errors = new_palette_errors
                self._palettes_accepted = True

                yield (new_total_image_error, output_4bit, line_to_palette,
                       new_palettes_rgb12_iigs, palettes_linear_rgb)

    def _propose_palettes(self) -> Tuple[np.ndarray, np.ndarray, List[float]]:
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
        using accept_palettes().
        """
        new_errors = list(self._errors)
        new_palettes_cam = np.empty_like(self._palettes_cam)
        new_palettes_rgb12_iigs = np.empty_like(self._palettes_rgb)

        # Compute a new 16-colour global palette for the entire image,
        # used as the starting center positions for k-means clustering of the
        # individual palettes
        self._fit_global_palette()

        self._mutate_palette_splits()
        for palette_idx in range(16):
            # print(palette_idx, self._palette_lines[palette_idx])
            # palette_lower, palette_upper = self._palette_splits[palette_idx]
            palette_pixels = (
                self._colours_cam[
                self._palette_lines[palette_idx], :, :].reshape(-1, 3))

            palettes_rgb12_iigs, palette_error = \
                dither_pyx.k_means_with_fixed_centroids(
                    n_clusters=16, n_fixed=self._reserved_colours,
                    samples=palette_pixels,
                    initial_centroids=self._global_palette,
                    max_iterations=1000, tolerance=0.05,
                    rgb12_iigs_to_cam16ucs=self._rgb12_iigs_to_cam16ucs
                )

            # if (palette_error >= self._errors[palette_idx] and not
            # self._reserved_colours):
            #     # Not a local improvement to the existing palette, so ignore it.
            #     # We can't take this shortcut when we're reserving colours
            #     # because it would break the invariant that all palettes must
            #     # share colours.
            #     continue
            for i in range(16):
                new_palettes_cam[palette_idx, i, :] = (
                    np.array(dither_pyx.convert_rgb12_iigs_to_cam(
                        self._rgb12_iigs_to_cam16ucs, palettes_rgb12_iigs[
                            i]), dtype=np.float32))

            new_palettes_rgb12_iigs[palette_idx, :, :] = palettes_rgb12_iigs
            new_errors[palette_idx] = palette_error

        self._palettes_accepted = False
        return new_palettes_cam, new_palettes_rgb12_iigs, new_errors

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
            dither_pyx.convert_cam16ucs_to_rgb12_iigs(
                clusters.cluster_centers_[frequency_order].astype(
                    np.float32)))

    def _mutate_palette_splits(self):
        if self._palettes_accepted:
            # Last time was good, keep going
            self._apply_palette_delta(self._palette_mutate_idx,
                                      self._palette_mutate_delta[0],
                                      self._palette_mutate_delta[1])
        else:
            # undo last mutation
            self._apply_palette_delta(self._palette_mutate_idx,
                                      -self._palette_mutate_delta[0],
                                      -self._palette_mutate_delta[1])

            # Pick a palette endpoint to move up or down
            palette_to_mutate = np.random.randint(0, 16)
            while True:
                if palette_to_mutate > 0:
                    palette_lower_delta = np.random.randint(-20, 21)
                else:
                    palette_lower_delta = 0
                if palette_to_mutate < 15:
                    palette_upper_delta = np.random.randint(-20, 21)
                else:
                    palette_upper_delta = 0
                if palette_lower_delta != 0 or palette_upper_delta != 0:
                    break

            self._apply_palette_delta(palette_to_mutate, palette_lower_delta,
                                      palette_upper_delta)

    def _apply_palette_delta(
            self, palette_to_mutate, palette_lower_delta, palette_upper_delta):
        old_lower, old_upper = self._palette_splits[palette_to_mutate]
        new_lower = old_lower + palette_lower_delta
        new_upper = old_upper + palette_upper_delta

        new_lower = np.clip(new_lower, 0, np.clip(new_upper, 1, 200) - 1)
        new_upper = np.clip(new_upper, new_lower + 1, 200)
        assert new_lower >= 0, new_upper - 1

        self._palette_splits[palette_to_mutate] = (new_lower, new_upper)
        self._palette_mutate_idx = palette_to_mutate
        self._palette_mutate_delta = (palette_lower_delta, palette_upper_delta)

    def _reassign_unused_palettes(self, new_line_to_palette, last_good_splits):
        palettes_used = [False] * 16
        for palette in new_line_to_palette:
            palettes_used[palette] = True
        best_palette_lines = [v for k, v in sorted(list(zip(
            self._palette_line_errors, range(200))))]

        # print(self._palette_lines)
        for palette_idx, palette_used in enumerate(palettes_used):
            if palette_used:
                continue
            # print("Reassigning palette %d" % palette_idx)

            # TODO: also remove from old entry
            worst_line = best_palette_lines.pop()
            self._palette_lines[palette_idx] = [worst_line]

            # print("Picked line %d with error %f" % (worst_line,
            #                                         self._palette_line_errors[worst_line]))


            #
            # worst_average_palette_error = 0
            # split_palette_idx = -1
            # idx = 0
            # for idx, lines in self._palette_lines.items():
            #     if len(lines) < 10:
            #         continue
            #     average_palette_error = np.sum(self._palette_line_errors[
            #                                     lines]) / len(lines)
            #     print(idx, average_palette_error)
            #     if average_palette_error > worst_average_palette_error:
            #         worst_average_palette_error = average_palette_error
            #         split_palette_idx = idx
            #
            # print("Picked %d with avg error %f" % (split_palette_idx, worst_average_palette_error))
            # # TODO: split off lines with largest error
            #
            # palette_line_errors = self._palette_line_errors[
            #     self._palette_lines[split_palette_idx]]
            #
            # print(sorted(
            #     list(zip(palette_line_errors, self._palette_lines[
            #     split_palette_idx])), reverse=True))
            # best_palette_lines = [v for k, v in sorted(
            #     list(zip(palette_line_errors, self._palette_lines[
            #     split_palette_idx])))]
            # num_max_lines = len(self._palette_lines[split_palette_idx])
            #
            # self._palette_lines[split_palette_idx] = best_palette_lines[
            #                                          :num_max_lines // 2]
            # # Move worst half to new palette
            # self._palette_lines[palette_idx] = best_palette_lines[
            #                                    num_max_lines // 2:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input image file to process.")
    parser.add_argument("output", type=str, help="Output file for converted "
                                                 "Apple II image.")
    parser.add_argument(
        "--lookahead", type=int, default=8,
        help=("How many pixels to look ahead to compensate for NTSC colour "
              "artifacts (default: 8)"))
    parser.add_argument(
        '--dither', type=str, choices=list(dither_pattern.PATTERNS.keys()),
        default=dither_pattern.DEFAULT_PATTERN,
        help="Error distribution pattern to apply when dithering (default: "
             + dither_pattern.DEFAULT_PATTERN + ")")
    parser.add_argument(
        '--show-input', action=argparse.BooleanOptionalAction, default=False,
        help="Whether to show the input image before conversion.")
    parser.add_argument(
        '--show-output', action=argparse.BooleanOptionalAction, default=True,
        help="Whether to show the output image after conversion.")
    parser.add_argument(
        '--palette', type=str, choices=list(set(palette_py.PALETTES.keys())),
        default=palette_py.DEFAULT_PALETTE,
        help='RGB colour palette to dither to.  "ntsc" blends colours over 8 '
             'pixels and gives better image quality on targets that '
             'use/emulate NTSC, but can be substantially slower.  Other '
             'palettes determine colours based on 4 pixel sequences '
             '(default: ' + palette_py.DEFAULT_PALETTE + ")")
    parser.add_argument(
        '--show-palette', type=str, choices=list(palette_py.PALETTES.keys()),
        help="RGB colour palette to use when --show_output (default: "
             "value of --palette)")
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction,
        default=False, help="Show progress during conversion")
    parser.add_argument(
        '--gamma_correct', type=float, default=2.4,
        help='Gamma-correct image by this value (default: 2.4)'
    )
    args = parser.parse_args()
    if args.lookahead < 1:
        parser.error('--lookahead must be at least 1')

    # palette = palette_py.PALETTES[args.palette]()
    screen = screen_py.SHR320Screen()

    # Conversion matrix from RGB to CAM16UCS colour values.  Indexed by
    # 24-bit RGB value
    rgb24_to_cam16ucs = np.load("data/rgb24_to_cam16ucs.npy")
    rgb12_iigs_to_cam16ucs = np.load("data/rgb12_iigs_to_cam16ucs.npy")

    # Open and resize source image
    image = image_py.open(args.input)
    if args.show_input:
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        srgb_output=False).show()
    rgb = np.array(
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        gamma=args.gamma_correct)).astype(np.float32) / 255

    # TODO: flags
    penalty = 1  # 1e18  # TODO: is this needed any more?
    iterations = 10 # 20

    pygame.init()
    # TODO: for some reason I need to execute this twice - the first time
    #  the window is created and immediately destroyed
    _ = pygame.display.set_mode((640, 400))
    canvas = pygame.display.set_mode((640, 400))
    canvas.fill((0, 0, 0))
    pygame.display.flip()

    total_image_error = None
    # TODO: reserved_colours should be a flag
    cluster_palette = ClusterPalette(
        rgb, reserved_colours=1,
        rgb12_iigs_to_cam16ucs=rgb12_iigs_to_cam16ucs,
        rgb24_to_cam16ucs=rgb24_to_cam16ucs)

    seq = 0
    for (new_total_image_error, output_4bit, line_to_palette,
         palettes_rgb12_iigs, palettes_linear_rgb) in cluster_palette.iterate(
        penalty, iterations):

        if total_image_error is not None:
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

        # dither = dither_pattern.PATTERNS[args.dither]()
        # bitmap = dither_pyx.dither_image(
        #     screen, rgb, dither, args.lookahead, args.verbose, rgb24_to_cam16ucs)

        # Show output image by rendering in target palette
        # output_palette_name = args.show_palette or args.palette
        # output_palette = palette_py.PALETTES[output_palette_name]()
        # output_screen = screen_py.DHGRScreen(output_palette)
        # if output_palette_name == "ntsc":
        #     output_srgb = output_screen.bitmap_to_image_ntsc(bitmap)
        # else:
        #     output_srgb = image_py.linear_to_srgb(
        #         output_screen.bitmap_to_image_rgb(bitmap)).astype(np.uint8)
        out_image = image_py.resize(
            Image.fromarray(output_srgb), screen.X_RES * 2, screen.Y_RES * 2,
            srgb_output=True)

        if args.show_output:
            surface = pygame.surfarray.make_surface(
                np.asarray(out_image).transpose((1, 0, 2)))  # flip y/x axes
            canvas.blit(surface, (0, 0))
            pygame.display.flip()

        seq += 1
        # Save Double hi-res image
        outfile = os.path.join(
            os.path.splitext(args.output)[0] + "-%d-preview.png" % seq)
        out_image.save(outfile, "PNG")
        screen.pack()
        # with open(args.output, "wb") as f:
        #     f.write(bytes(screen.aux))
        #     f.write(bytes(screen.main))
        with open(args.output, "wb") as f:
            f.write(bytes(screen.memory))

    # print((palettes_rgb * 255).astype(np.uint8))
    unique_colours = np.unique(
        palettes_rgb12_iigs.reshape(-1, 3), axis=0).shape[0]
    print("%d unique colours" % unique_colours)



if __name__ == "__main__":
    main()
