# ][-pix

][-pix is an image conversion utility targeting Apple II graphics modes, currently Double Hi-Res.

## Installation

Requires:
*  python 3.x
*  numpy
*  cython
*  colour-science [XXX]

XXX cython compilation

XXX precompute distance matrix

## Examples
Original:

![Two colourful parrots sitting on a branch](examples/Vibrant_Wings.jpg)

Preview:

![Two colourful parrots sitting on a branch](examples/wings-preview.png)

OpenEmulator screenshot:

![Two colourful parrots sitting on a branch](examples/wings-openemulator.png)

 (Source: [Wikimedia](https://commons.wikimedia.org/wiki/File:Vibrant_Wings.jpg))


## Some background on Apple II Double Hi-Res graphics

Like other (pre-//gs) Apple II graphics modes, Double Hi-Res relies on NTSC Artifact Colour, which means that the colour of a pixel is entirely determined by its horizontal position on the screen, and the on/off status of preceding horizontal pixels.

In Double Hi-Res mode, there are 560 horizontal pixels per line, which are individually addressable.  This is an improvement over the (single) Hi-Res mode, which also has 560 horizontal pixels, but which can only be addressed in groups of two (with an option to shift a block of 3.5 pixels by one dot).  See XXX for an introduction to this.

Double Hi-Res is capable of producing 16 display colours, but with heavy restrictions on how these colours can be arranged horizontally.  One simple model is to only treat the display in groups of 4 horizontal pixels, which gives an effective resolution of 140x192 in 16 colours.  These 140 pixel colours can be chosen independently, but they exhibit interference/fringing effects when two colours meet.

A more useful model for thinking about DHGR is to consider a _sliding_ window of 4 horizontal pixels moving across the screen.   These 4 pixel values produce one of 16 colours at each of 560 horizontal positions.  The precise mapping depends also on the value x%4, which approximates the NTSC colour phase.

This allows us to understand and predict the interference behaviour in terms of the "effective 140px" model described above.

XXX slides

When we consider the next horizontal position, 3 of the values in our sliding window are already fixed, and we only have 2 choices available, namely a 0 or 1 in the new position.  This means that there are only *two* possible colours for each successive pixel.  One of these corresponds to the same colour as the current pixel; the other is some other colour from our palette of 16.

So, if we want to transition from one colour to a particular new colour, it may take up to 4 horizontal pixels before we are able to achieve it (e.g. 0000 --> 1111).  In the meantime we have to transition through up to 3 other colours, which may or may not be desirable visually.

These constraints are difficult to work with when constructing DHGR graphics "by hand" (though you can easily construct a transition graph/table showing the available choices for a given x colour and x%4 value), but we can account for them programmatically in our image conversion to take full advantage of the "true" 560px resolution while accounting for colour interference effects.

### Limitations of this colour model

In practise the above description of the Apple II colour model is only a discrete approximation.  On real hardware, the video signal is a continuous analogue signal, and colour is continuously modulated rather than producing discrete coloured pixels with fixed colour values.

Furthermore, in an NTSC video signal the colour (chroma) signal has a lower bandwidth than the luma (brightness) signal,
which means that colours will tend to bleed across multiple pixels.  i.e. the influence on pixel x+1 from previous pixel on/off states is a more complex function then the mapping described above. 

This means that images produced by ][-pix do not always quite match colours produced on real hardware (or high-fidelity emulators, like OpenEmulator) due to this colour bleeding effect.  In principle, it would be possible to simulate the NTSC video signal more directly to account for this during image processing.

For example XXX

## Dithering and Double Hi-Res

Dithering an image to produce an approximation with fewer image colours is a well-known technique.  The basic idea is to pick a "best colour match" for a pixel from our limited palette, then to compute the difference between the true and selected colour values and diffuse this error to nearby pixels (using some pattern).

In the particular case of DHGR this algorithm runs into difficulties, because each pixel only has two possible colour choices (from a total of 16).  If we only consider the two possibilities for the immediate next pixel then neither may be a particularly good match.  However it may be more beneficial to make a suboptimal choice now (deliberately introduce more error), if it allows us access to a better colour for a subsequent pixel.  "Classical" dithering algorithms do not account for these palette constraints, and produce suboptimal image quality for DHGR conversions. 

We can deal with this by looking ahead N pixels (6 by default) for each image position (x,y), and computing the effect of choosing all 2^N combinations of these N-pixel states on the dithered source image.

Specifically, for a fixed choice of one of these N pixel sequences, we tentatively perform the error diffusion as normal on a copy of the image, and compute the total mean squared distance from the (fixed) N-pixel sequence to the error-diffused source image.  For the perceptual colour distance metric we use CIE2000 delta-E, see XXX

Finally, we pick the N-pixel sequence with the lowest total error, and select the first pixel of this N-pixel sequence for position (x,y).  We then performing error diffusion as usual for this single pixel, and proceed to x+1.

This allows us to "look beyond" local minima to find cases where it is better to make a suboptimal choice now to allow better overall image quality in subsequent pixels.  Since we will sometimes find that our choice of 2 next-pixel colours actually includes (or comes close to) the "ideal" choice, this means we can take maximal advantage of the 560-pixel horizontal resolution.

## Palettes

Since the Apple II graphics are not based on RGB colour, we have to approximate an RGB colour palette when dithering an RGB image.

Different emulators have made (often quite different) choices for their RGB colour palettes, so an image that looks good on one emulator may not look good on another (or on real hardware).  For example Virtual II uses two different RGB shades of grey for the DHGR colour values that are rendered as identical shade of grey in NTSC.

Secondly, the actual display colours rendered by an Apple II are not fixed, but bleed into each other due to the behaviour of the (analogue) NTSC video signal.  i.e. the entire notion of a "16-colour RGB palette" is a flawedone.  The model described above where we can assign from 16 fixed colours to each of 560 discrete pixels is only an approximation (though a useful one in practise).

Some emulators emulate the NTSC video signal more faithfully (e.g. OpenEmulator), in which case they do not have a true "RGB palette".  Others (e.g. Virtual II) seem to use a discrete approximation similar to the one described above, so a fixed palette can be reconstructed.

To compute the emulator palettes used by ][-pix I measured the sRGB colour values produced by a full-screen Apple II colour image (using the colour picker tool of Mac OS X).  I have not yet attempted to measure/estimate palettes of other emulators, or "real hardware" (since I don't actually have a composite colour monitor!)

Existing conversion tools (see below) tend to support a variety of RGB palette values sourced from various places (older tools, emulators, theoretical estimations etc).  I suppose the intention is to try various of these on your target platform (emulator or hardware) to see which give good results. In practise I think it would be more useful to only support additional targets that are in modern use.

## Precomputing distance matrix

Computing the CIE2000 distance between two colour values is fairly expensive, since the formula is complex. We deal with this by precomputing a matrix from all 256^3 integer RGB values to the 16 RGB values in a palette. This 256MB matrix is generated on disk by the precompute_distance.py utility, and is mmapped at runtime for efficient access.

# Comparison to other DHGR image converters

## bmp2dhr

*  bmp2dhr (XXX) supports additional graphics modes not yet supported by ii-pix, namely (double) lo-res, and hi-res.  Support for the lores modes would be easy to add to ii-pix, although hi-res requires more work to accommodate the colour model.  A similar lookahead strategy will likely work well though.

*  supports additional dither modes

*  It does not perform RGB colour space conversions before dithering, i.e. if the input image is in sRGB colour space (as most digital images will be) then the dithering is also performed in sRGB.  Since sRGB is not a linear colour space, the effect of dithering is to distribute errors non-linearly, which reduces image quality.

*  DHGR conversions are treated as simple 140x192x16 colour images without colour constraints, and ignores the colour fringing behaviour described above.  The generated .bmp preview images also do not show fringing, but it is (of course) present when viewing the image on an Apple II or emulator that accounts for it.  i.e. the preview images are not especially representative of the actual results.

*  Apart from ignoring DHGR colour interactions, the 140px converted images are also lower than ideal resolution since they do not make use of the ability to address all 560px independently.

*  The perceptual colour distance metric used to match the best colour to an input pixel seems to be an ad-hoc one based on a weighted sum of Euclidean sRGB distance and Rec.601 luma value.  In practise this seems to give lower quality results than CIE2000 (though the latter is slower to compute - which is why we precompute the distance matrix ahead of time)

## a2bestpix ([Link](http://lukazi.blogspot.com/2017/03/double-high-resolution-graphics-dhgr.html))

*  Like ii-pix, it only supports DHGR conversion.  Overall quality is fairly good although colours are slightly distorted (for reasons described below), and the generated preview images do not quite give a faithful representation of the native image quality.  

*  Like ii-pix, and unlike bmp2dhr, a2bestpix does apply a model of the DHGR colour interactions, albeit an ad-hoc one based on rules and tables of 4-pixel "colour blocks" reconstructed from (AppleWin) emulator behaviour.  This does allow it to make use of (closer to) full 560px resolution, although it still treats the screen as a sequence of 140 4-pixel colour blocks (with some constraints on the allowed arrangement of these blocks).

*  supports additional dither modes (partly out of necessity due to the custom "colour block" model)

*  Supports a variety of perceptual colour distance metrics including CIE2000 and the one bmp2dhr uses.  In practise I'm not sure the others are useful since CIE2000 is the most recent refinement of much research on this topic, and is the most accurate.

*  Does not transform from sRGB to linear RGB before dithering (though sRGB conversion is done when computing CIE2000 distance), so error is diffused non-linearly.  This impacts colour balance when dithering.

*  image conversion performs an optimization over groups of multiple pixels (via choice of "colour blocks").  From what I can tell this minimizes the total colour distance from a fixed list of colour blocks to a group of 4 target pixels, similar to --lookahead=4 for ii-pix (though I'm not sure it's evaluating all 2^4 pixel combinations).  But since the image is (AFAICT) treated as a sequence of (non-overlapping) 4-pixel blocks this does not result in optimizing each output pixel independently.

*  The list of "colour blocks" seem to contain colour sequences that cannot actually be rendered on the Apple II.  For example compare the spacing of yellow and orange pixels on the parrot between the preview image (LHS) and openemulator (RHS): 

![Detail of a2bestpix preview image](docs/a2bestbix-preview-crop.png)
![Detail of openemulator render](docs/a2bestpix-openemulator-crop.png)

*  Other discrepancies are also visible when comparing these two images.  This means that (like bmp2dhr) the generated "preview" image does not closely match the native image, and the dithering algorithm is also optimizing over a slightly incorrect set of colour sequences, which impacts image quality.  Possibly these are transcription errors, or artifacts of the particular emulator (AppleWin) from which they were reconstructed.

# Future work

* Supporting lo-res and double lo-res graphics modes would be straightforward.

* Hi-res will require more care, since the 560 pixel display is not individually dot addressible.  In particular the behaviour of the "palette bit" (which shifts a group of 7 dots to the right by 1) is another optimization constraint.  In practise a similar lookahead algorithm should work well though.

* With more work to model the NTSC video signal it should be possible to produce images that better account for the NTSC signal behaviour.  For example I think it is still true that at each horizontal dot position there is a choice of two possible "output colours", but these are influenced by the previous pixel values in a more complex way and do not come from a fixed palette of 16 choices.

* I would like to be able to find an ordered dithering algorithm that works well for Apple II graphics.  Ordered dithering specifically avoids diffusing errors arbitrarily across the image, which produces visual noise (and unnecessary deltas) when combined with animation.  For example such a thing may work well with my II-Vision video streamer.  However the properties of NTSC artifact colour seem to be in conflict with these requirements, i.e. pixel changes *always* propagate colour to some extent.
