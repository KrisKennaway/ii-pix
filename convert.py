"""Image converter to Apple II Double Hi-Res format."""

import argparse
import array
import os.path
import time
import collections
import random
import pygame

import colour
from PIL import Image
import numpy as np
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn import cluster

import dither as dither_pyx
import dither_pattern
import image as image_py
import palette as palette_py
import screen as screen_py


# TODO:
# - support LR/DLR
# - support HGR

class ClusterPalette:
    def __init__(self, image: Image):
        self._colours_cam = self._image_colours_cam(image)
        self._best_palette_distances = {i: (1e9, None) for i in range(16)}
        self._iterations = 0
        self._palettes_cam = np.empty((16, 16, 3), dtype=np.float32)
        self._palettes_rgb = np.empty((16, 16, 3), dtype=np.float32)
        self._global_palette = self._fit_global_palette()

    def _image_colours_cam(self, image: Image):
        colours_rgb = np.asarray(image).reshape((-1, 3))
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            colours_cam = colour.convert(colours_rgb, "RGB",
                                         "CAM16UCS").astype(np.float32)
        return colours_cam

    def _fit_global_palette(self):
        """Compute a 16-colour palette for the entire image to use as
        starting point for the sub-palettes.  This should help when the image
        has large blocks of colour since the sub-palettes will tend to pick the same coloursx."""
        clusters = cluster.MiniBatchKMeans(n_clusters=16, max_iter=10000)
        # tol=0.0000000001, algorithm="elkan")
        clusters.fit_predict(self._colours_cam)
        return clusters.cluster_centers_

    def iterate(self):
        self._iterations += 1
        print("Iteration %d" % self._iterations)
        for palette_idx in range(16):


            # i=5: 3 * (200/16) : 7 * (200/16)
            # print("Fitting palette %d" % palette_idx)
            p_lower2 = max(palette_idx - 1.5, 0)
            p_lower1 = max(palette_idx - 1, 0)
            p_lower0 = palette_idx
            p_upper0 = max(palette_idx + 1, 16)
            p_upper1 = max(palette_idx + 2, 16)
            p_upper2 = min(palette_idx + 2.5, 16)
            # TODO: weight +/-1 and 0 bands higher
            # TODO: dynamically tune palette cuts
            palette_pixels = np.concatenate(
                [
                    self._colours_cam[
                        int(p_lower2 * (200 / 16)) * 320:int(p_upper2 * (
                            200 / 16)) * 320, :],
                    # self._colours_cam[
                    #      int(p_lower1 * (200 / 16)) * 320:int(p_upper1 * (
                    #          200 / 16)) * 320, :],
                    # self._colours_cam[
                    #      int(p_lower0 * (200 / 16)) * 320:int(p_upper0 * (
                    #          200 / 16)) * 320, :],
                ], axis=0)

            best_wce, best_medians = self._best_palette_distances[palette_idx]
            # if palette_idx == 0:
            # initial_centers = kmeans_plusplus_initializer(
            #     palette_pixels, 16).initialize()
            # else:
            #     initial_centers = kmedians_instance.get_medians()

            # kmedians_instance = kmeans(
            #     palette_pixels, initial_centers, tolerance=0.0000000001,
            #     itermax=100,
            #     metric=distance_metric(type_metric.EUCLIDEAN_SQUARE))
            # kmedians_instance.process()
            # TODO: tolerance
            clusters = cluster.MiniBatchKMeans(
                n_clusters=16, max_iter=10000, init=self._global_palette,
                n_init=1)
            # tol=0.0000000001, algorithm="elkan")
            clusters.fit_predict(palette_pixels)
            # if kmedians_instance.get_total_wce() < best_wce:
            #    best_wce = kmedians_instance.get_total_wce()
            #    best_medians = kmedians_instance
            if clusters.inertia_ < (best_wce * 0.99):
                best_wce = clusters.inertia_
                print("Improved palette %d: %f" % (palette_idx, best_wce))

                # self._palettes_cam[palette_idx, :, :] = np.array(
                #     best_medians.get_centers()).astype(np.float32)

                self._palettes_cam[palette_idx, :, :] = np.array(
                    clusters.cluster_centers_).astype(np.float32)
                self._best_palette_distances[palette_idx] = (
                    best_wce, best_medians)

                with colour.utilities.suppress_warnings(
                        colour_usage_warnings=True):
                    palette_rgb = colour.convert(
                        self._palettes_cam[palette_idx], "CAM16UCS", "RGB")
                    # SHR colour palette only uses 4-bit values
                    palette_rgb = np.round(palette_rgb * 15) / 15
                    self._palettes_rgb[palette_idx, :, :] = palette_rgb.astype(
                        np.float32)

        return self._palettes_cam, self._palettes_rgb


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
    rgb_to_cam16 = np.load("data/rgb_to_cam16ucs.npy")

    # Open and resize source image
    image = image_py.open(args.input)
    if args.show_input:
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        srgb_output=False).show()
    rgb = np.array(
        image_py.resize(image, screen.X_RES, screen.Y_RES,
                        gamma=args.gamma_correct, srgb_output=True)).astype(
        np.float32) / 255

    penalty = 10  # 1e9
    iterations = 50

    pygame.init()
    canvas = pygame.display.set_mode((640, 400))
    canvas = pygame.display.set_mode((640, 400))
    canvas.fill((0, 0, 0))
    pygame.display.flip()
    # print("Foo")

    cluster_palette = ClusterPalette(rgb)
    for iteration in range(iterations):
        palettes_cam, palettes_rgb = cluster_palette.iterate()
        # print((palettes_rgb*255).astype(np.uint8))
        for i in range(16):
            screen.set_palette(i, (np.round(palettes_rgb[i, :, :] * 15)).astype(
                np.uint8))

        output_4bit, line_to_palette = dither_pyx.dither_shr(
            rgb, palettes_cam, palettes_rgb, rgb_to_cam16, float(penalty))
        screen.set_pixels(output_4bit)
        output_rgb = np.zeros((200, 320, 3), dtype=np.uint8)
        for i in range(200):
            screen.line_palette[i] = line_to_palette[i]
            output_rgb[i, :, :] = (
                    palettes_rgb[line_to_palette[i]][
                        output_4bit[i, :]] * 255).astype(np.uint8)
        output_srgb = image_py.linear_to_srgb(output_rgb).astype(np.uint8)

        # dither = dither_pattern.PATTERNS[args.dither]()
        # bitmap = dither_pyx.dither_image(
        #     screen, rgb, dither, args.lookahead, args.verbose, rgb_to_cam16)

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
            srgb_output=False)  # XXX true

        if args.show_output:
            surface = pygame.surfarray.make_surface(np.asarray(
                out_image).transpose((1, 0, 2)))
            canvas.blit(surface, (0, 0))
            pygame.display.flip()

        # Save Double hi-res image
        outfile = os.path.join(os.path.splitext(args.output)[0] +
                               "-%d-preview.png" % cluster_palette._iterations)
        out_image.save(outfile, "PNG")
        screen.pack()
        # with open(args.output, "wb") as f:
        #     f.write(bytes(screen.aux))
        #     f.write(bytes(screen.main))
        with open("%s-%s" % (args.output, cluster_palette._iterations),
                  "wb") as f:
            f.write(bytes(screen.memory))


if __name__ == "__main__":
    main()
