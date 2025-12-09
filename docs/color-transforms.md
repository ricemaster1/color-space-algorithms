# Color Transforms

Color transforms remap source RGB pixels into alternate spaces before quantization or dithering. Use them to emphasize luminance, preserve hue ordering, or bridge into perceptual Delta E formulas.

## Workflow
1. Run the transform script to emit an intermediate PNG or `.s` file (depending on the script's default behavior).
2. Feed the transformed output into a distance metric or quantizer that expects that space (e.g., Lab for Delta E 2000).
3. Finish with a dither if you need additional texture control.

## Modules

### rgb_to_lab.py
Converts RGB to CIE Lab. Optional weights prioritize L, a, or b.
```bash
python algorithms/color_transforms/rgb_to_lab.py forest.png --weights 1,1,0.2 -o forest_lab.png
```
Parameters: `image`, `-o/--output`, `--weights` (comma separated L,a,b weights, default `1,1,1`).

### rgb_to_hsv_hsl.py
Exports HSV or HSL channels with custom weighting.
```bash
python algorithms/color_transforms/rgb_to_hsv_hsl.py art.png --space hsl --weights 1,0.5,0.5 -o art_hsl.png
```
Parameters: `image`, `-o/--output`, `--space` (`hsv` or `hsl`, default `hsv`), `--weights` (weights for hue, saturation, value/lightness).
Deep dive (theory, weights, images): `docs/scripts/rgb_to_hsv_hsl.md`.

### rgb_to_xyz.py
Bridges RGB to XYZ, a prerequisite for Lab or YCbCr conversions.
```bash
python algorithms/color_transforms/rgb_to_xyz.py studio.png --weights 0.8,1.0,0.6 -o studio_xyz.png
```
Parameters: `image`, `-o/--output`, `--weights` (X,Y,Z weights, default `1,1,1`).

### rgb_to_ycbcr.py
Produces Y (luma) plus Cb/Cr (chroma) data for luma-focused dithers.
```bash
python algorithms/color_transforms/rgb_to_ycbcr.py film.png --weights 1.2,1,1 -o film_ycbcr.png
```
Parameters: `image`, `-o/--output`, `--weights` (Y,Cb,Cr weights, default `1,1,1`).

Design your own pipelines by chaining these scripts in front of quantizers (e.g., Lab transform → Delta E → K-means → Sierra dither) to dial in the visual style you need.
