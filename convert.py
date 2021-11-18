"""Image converter to Apple II Double Hi-Res format."""

import argparse
import os.path
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
            self, image: Image, rgb12_iigs_to_cam16ucs, reserved_colours=0):
        self._colours_cam = self._image_colours_cam(image)
        self._reserved_colours = reserved_colours
        self._errors = [1e9] * 16
        self._palettes_cam = np.empty((16, 16, 3), dtype=np.float32)
        self._palettes_rgb = np.empty((16, 16, 3), dtype=np.uint8)
        self._global_palette = np.empty((16, 16, 3), dtype=np.float32)
        self._rgb12_iigs_to_cam16ucs = rgb12_iigs_to_cam16ucs

    def _image_colours_cam(self, image: Image):
        colours_rgb = np.asarray(image).reshape((-1, 3))
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            colours_cam = colour.convert(colours_rgb, "RGB",
                                         "CAM16UCS").astype(np.float32)
        return colours_cam

    def _fit_global_palette(self):
        """Compute a 16-colour palette for the entire image to use as
        starting point for the sub-palettes.  This should help when the image
        has large blocks of colour since the sub-palettes will tend to pick the
        same colours."""

        clusters = cluster.MiniBatchKMeans(n_clusters=16, max_iter=10000)
        clusters.fit_predict(self._colours_cam)

        labels = clusters.labels_
        frequency_order = [
            k for k, v in sorted(
                # List of (palette idx, frequency count)
                list(zip(*np.unique(labels, return_counts=True))),
                key=lambda kv: kv[1], reverse=True)]

        res = dither_pyx.convert_cam16ucs_to_rgb12_iigs(
                clusters.cluster_centers_[frequency_order].astype(
                    np.float32))
        return res

    def propose_palettes(self) -> Tuple[np.ndarray, np.ndarray, List[float]]:
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
        new_palettes_cam = np.copy(self._palettes_cam)
        new_palettes_rgb12_iigs = np.copy(self._palettes_rgb)

        # Compute a new 16-colour global palette for the entire image,
        # used as the starting center positions for k-means clustering of the
        # individual palettes
        self._global_palette = self._fit_global_palette()

        dynamic_colours = 16 - self._reserved_colours

        # The 16 palettes are striped across consecutive (overlapping) line
        # ranges.  The basic unit is 200/16 = 12.5 lines, but we extend the
        # line range to cover a multiple of this so that the palette ranges
        # overlap.  Since nearby lines tend to have similar colours, this has
        # the effect of smoothing out the colour transitions across palettes.
        palette_band_width = 3
        for palette_idx in range(16):
            p_lower = max(palette_idx + 0.5 - (palette_band_width / 2), 0)
            p_upper = min(palette_idx + 0.5 + (palette_band_width / 2), 16)
            # TODO: dynamically tune palette cuts
            palette_pixels = self._colours_cam[
                             int(p_lower * (200 / 16)) * 320:int(p_upper * (
                                     200 / 16)) * 320, :]

            # TODO: tune tolerance
            # clusters = cluster.MiniBatchKMeans(
            #     n_clusters=16, max_iter=10000,
            #     init=self._global_palette,
            #     n_init=1)
            # clusters.fit_predict(palette_pixels)
            #
            # palette_error = clusters.inertia_

            palettes_rgb12_iigs, palette_error = \
                dither_pyx.k_means_with_fixed_centroids(
                    n_clusters=16, n_fixed=self._reserved_colours,
                    samples=palette_pixels,
                    initial_centroids=self._global_palette,
                    max_iterations=1000, tolerance=0.05,
                    rgb12_iigs_to_cam16ucs=self._rgb12_iigs_to_cam16ucs
                )

            if (palette_error >= self._errors[palette_idx] and not
            self._reserved_colours):
                # Not a local improvement to the existing palette, so ignore it.
                # We can't take this shortcut when we're reserving colours
                # because it would break the invariant that all palettes must
                # share colours.
                continue
            for i in range(16):
                new_palettes_cam[palette_idx, i, :] = (
                    np.array(dither_pyx.convert_rgb12_iigs_to_cam(
                        self._rgb12_iigs_to_cam16ucs, palettes_rgb12_iigs[
                            i]), dtype=np.float32))

            # Suppress divide by zero warning,
            # https://github.com/colour-science/colour/issues/900
            # with colour.utilities.suppress_warnings(python_warnings=True):
            #     palette_rgb = colour.convert(
            #         new_palettes_cam[palette_idx, :, :], "CAM16UCS", "RGB")
            # palette_rgb_rec601 = np.clip(image_py.srgb_to_linear(
            #     colour.YCbCr_to_RGB(
            #         colour.RGB_to_YCbCr(
            #             image_py.linear_to_srgb(palette_rgb * 255) / 255,
            #             K=colour.WEIGHTS_YCBCR['ITU-R BT.709']),
            #         K=colour.WEIGHTS_YCBCR['ITU-R BT.601']) * 255) / 255, 0, 1)
            # palette_rgb = np.clip(
            #     image_py.srgb_to_linear(
            #         colour.YCbCr_to_RGB(
            #             colour.RGB_to_YCbCr(
            #                 image_py.linear_to_srgb(
            #                     palette_rgb[:, :] * 255) / 255,
            #                 K=colour.WEIGHTS_YCBCR['ITU-R BT.709']),
            #             K=colour.WEIGHTS_YCBCR[
            #                 'ITU-R BT.601']) * 255) / 255,
            #     0, 1)
            new_palettes_rgb12_iigs[palette_idx, :, :] = palettes_rgb12_iigs
            new_errors[palette_idx] = palette_error

        return new_palettes_cam, new_palettes_rgb12_iigs, new_errors

    def accept_palettes(
            self, new_palettes_cam: np.ndarray,
            new_palettes_rgb: np.ndarray, new_errors: List[float]):
        self._palettes_cam = np.copy(new_palettes_cam)
        self._palettes_rgb = np.copy(new_palettes_rgb)
        self._errors = list(new_errors)


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
    penalty = 1e9
    iterations = 50

    pygame.init()
    # TODO: for some reason I need to execute this twice - the first time
    #  the window is created and immediately destroyed
    _ = pygame.display.set_mode((640, 400))
    canvas = pygame.display.set_mode((640, 400))
    canvas.fill((0, 0, 0))
    pygame.display.flip()

    total_image_error = 1e9
    iterations_since_improvement = 0

    # palettes_iigs = np.empty((16, 16, 3), dtype=np.uint8)
    cluster_palette = ClusterPalette(
        rgb, reserved_colours=1, rgb12_iigs_to_cam16ucs=rgb12_iigs_to_cam16ucs)

    while iterations_since_improvement < iterations:
        new_palettes_cam, new_palettes_rgb12_iigs, new_palette_errors = (
            cluster_palette.propose_palettes())

        # Suppress divide by zero warning,
        # https://github.com/colour-science/colour/issues/900
        with colour.utilities.suppress_warnings(python_warnings=True):
            new_palettes_linear_rgb = colour.convert(
                new_palettes_cam, "CAM16UCS", "RGB").astype(np.float32)

        # Recompute image with proposed palettes and check whether it has
        # lower total image error than our previous best.
        new_output_4bit, new_line_to_palette, new_total_image_error = \
            dither_pyx.dither_shr(
                rgb, new_palettes_cam, new_palettes_linear_rgb,
                rgb24_to_cam16ucs, float(penalty))
        if new_total_image_error >= total_image_error:
            iterations_since_improvement += 1
            continue

        # We found a globally better set of palettes
        iterations_since_improvement = 0
        cluster_palette.accept_palettes(
            new_palettes_cam, new_palettes_rgb12_iigs, new_palette_errors)

        if total_image_error < 1e9:
            print("Improved quality +%f%% (%f)" % (
                (1 - new_total_image_error / total_image_error) * 100,
                new_total_image_error))
        output_4bit = new_output_4bit
        line_to_palette = new_line_to_palette
        total_image_error = new_total_image_error
        palettes_rgb12_iigs = new_palettes_rgb12_iigs
        palettes_linear_rgb = new_palettes_linear_rgb
        # # Recompute 4-bit //gs RGB palettes
        # palette_rgb_rec601 = np.clip(
        #     colour.YCbCr_to_RGB(
        #         colour.RGB_to_YCbCr(
        #             image_py.linear_to_srgb(palettes_rgb12_iigs * 255) / 255,
        #             K=colour.WEIGHTS_YCBCR['ITU-R BT.709']),
        #         K=colour.WEIGHTS_YCBCR['ITU-R BT.601']), 0, 1)
        #
        # palettes_iigs = np.round(palette_rgb_rec601 * 15).astype(np.uint8)
        for i in range(16):
            screen.set_palette(i, palettes_rgb12_iigs[i, :, :])

        # Recompute current screen RGB image
        screen.set_pixels(output_4bit)
        output_rgb = np.empty((200, 320, 3), dtype=np.uint8)
        for i in range(200):
            screen.line_palette[i] = line_to_palette[i]
            output_rgb[i, :, :] = (
                    palettes_linear_rgb[line_to_palette[i]][output_4bit[i, :]] * 255
            ).astype(
                # np.round(palettes_rgb[line_to_palette[i]][
                #              output_4bit[i, :]] * 15) / 15 * 255).astype(
                np.uint8)
        # output_srgb_rec709 = np.clip(colour.YCbCr_to_RGB(
        #     colour.RGB_to_YCbCr(
        #         image_py.linear_to_srgb(output_rgb) / 255,
        #         K=colour.WEIGHTS_YCBCR['ITU-R BT.601']),
        #     K=colour.WEIGHTS_YCBCR['ITU-R BT.709']), 0, 1) * 255

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
    # print((palettes_rgb * 255).astype(np.uint8))
    unique_colours = np.unique(palettes_rgb12_iigs.reshape(-1, 3), axis=0).shape[0]
    print("%d unique colours" % unique_colours)

    # Save Double hi-res image
    outfile = os.path.join(os.path.splitext(args.output)[0] + "-preview.png")
    out_image.save(outfile, "PNG")
    screen.pack()
    # with open(args.output, "wb") as f:
    #     f.write(bytes(screen.aux))
    #     f.write(bytes(screen.main))
    with open(args.output, "wb") as f:
        f.write(bytes(screen.memory))


if __name__ == "__main__":
    main()
