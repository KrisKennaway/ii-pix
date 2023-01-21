"""Precompute CAM16-UCS colour tuples for all 2^24 RGB tuples.

This 192MB data file is used to convert from RGB to CAM16-UCS colour space
for purposes of computing (approximate) perceptual difference between pairs of
colours when optimizing the image conversion (since this perceptual
difference corresponds to the Euclidean distance in this colour space)
"""

import colour
import numpy as np


def srgb_to_linear_rgb_array(a: np.ndarray, gamma=2.4) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** gamma)


def main():
    print("Precomputing conversion matrix from 24-bit RGB to CAM16UCS colour "
          "space")
    # Compute matrix of all 24-bit RGB values, normalized to 0..1 range
    bits24 = np.arange(2 ** 24, dtype=np.uint32).reshape(-1, 1)
    all_rgb24 = np.concatenate(
        [bits24 >> 16 & 0xff, bits24 >> 8 & 0xff, bits24 & 0xff],
        axis=1).astype(np.float32) / 255
    del bits24

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        # Compute matrix of corresponding CAM16UCS colour values, indexed
        # by 24-bit RGB value
        rgb24_to_cam16ucs = colour.convert(all_rgb24, "RGB", "CAM16UCS").astype(
            np.float32)
    del all_rgb24
    np.save("data/rgb24_to_cam16ucs.npy", np.ascontiguousarray(
        rgb24_to_cam16ucs))
    del rgb24_to_cam16ucs

    print("Precomputing conversion matrix from 12-bit //gs RGB to CAM16UCS "
          "colour space")
    # Compute matrix of all 12-bit RGB values, normalized to 0..1 range
    bits12 = np.arange(2 ** 12, dtype=np.uint32).reshape(-1, 1)
    r = bits12 >> 8
    g = (bits12 >> 4) & 0xf
    b = bits12 & 0xf
    all_rgb12 = np.concatenate(
        [(r << 4) | r, (g << 4) | g, (b << 4) | b], axis=1).astype(
        np.float32) / 255
    del bits12, r, g, b

    # //gs RGB values use gamma-corrected Rec.601 RGB colour space.  We need to
    # convert to Rec.709 RGB as preparation for converting to CAM16UCS.  We
    # do this via the YCbCr intermediate color model.
    rgb12_iigs = np.clip(srgb_to_linear_rgb_array(
        np.clip(colour.YCbCr_to_RGB(
            colour.RGB_to_YCbCr(
                all_rgb12, K=colour.WEIGHTS_YCBCR[
                    'ITU-R BT.601']),
            K=colour.WEIGHTS_YCBCR['ITU-R BT.709']), 0, 1)), 0, 1)
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        # Compute matrix of corresponding CAM16UCS colour values, indexed
        # by 12-bit //gs RGB value
        rgb12_iigs_to_cam16ucs = colour.convert(
            rgb12_iigs, "RGB", "CAM16UCS").astype(np.float32)
    del rgb12_iigs
    np.save("data/rgb12_iigs_to_cam16ucs.npy", np.ascontiguousarray(
        rgb12_iigs_to_cam16ucs))
    del rgb12_iigs_to_cam16ucs


if __name__ == "__main__":
    main()
