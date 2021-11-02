"""Precompute CAM16-UCS colour tuples for all 2^24 RGB tuples.

This 192MB data file is used to convert from RGB to CAM16-UCS colour space
for purposes of computing (approximate) perceptual difference between pairs of
colours when optimizing the image conversion (since this perceptual
difference corresponds to the Euclidean distance in this colour space)
"""

import colour
import numpy as np


def main():
    print("Precomputing conversion matrix from RGB to CAM16UCS colour space")

    # Compute matrix of all 24-bit RGB values, normalized to 0..1 range
    bits24 = np.arange(2 ** 24, dtype=np.uint32).reshape(-1, 1)
    all_rgb = np.concatenate(
        [bits24 >> 16 & 0xff, bits24 >> 8 & 0xff, bits24 & 0xff],
        axis=1).astype(np.float32) / 255
    del bits24

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        # Compute matrix of corresponding CAM16UCS colour values, indexed
        # by 24-bit RGB value
        all_cam16 = colour.convert(all_rgb, "RGB", "CAM16UCS").astype(
            np.float32)
    del all_rgb
    np.save("data/rgb_to_cam16ucs.npy", all_cam16)


if __name__ == "__main__":
    main()
