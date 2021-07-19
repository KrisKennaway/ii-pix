"""Precompute CIE2000 perceptual colour distance matrices.

The matrix of delta-E values is computed for all pairs of 24-bit RGB values,
and Apple II target palette values.  This is written out as a file that is
mmapped at runtime for efficient access.  For a 16-colour target palette this
file is 256MB; for a 256-colour (NTSC) target palette it is 4GB.
"""

import colour
import numpy as np


def main():
    print("Precomputing conversion matrix from RGB to CAM16UCS colour space")

    # Compute matrix of all 24-bit RGB values, normalized to 0..1 range
    bits24 = np.arange(2 ** 24).reshape(-1, 1)
    all_rgb = (np.concatenate(
        [bits24 >> 16 & 0xff, bits24 >> 8 & 0xff, bits24 & 0xff],
        axis=1) / 255).astype(np.float32)

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        # Compute matrix of corresponding CAM16UCS colour values, indexed
        # by 24-bit RGB value
        all_cam16 = colour.convert(all_rgb, "RGB", "CAM16UCS").astype(
            np.float32)
    np.save("data/rgb_to_cam16ucs.npy", all_cam16)


if __name__ == "__main__":
    main()
