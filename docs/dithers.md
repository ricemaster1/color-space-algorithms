# Dithers

Error diffusion is the final pass that determines how smooth gradients look once pixels are locked to ARMLite's palette. These renderers assume you already quantized colors (or you let the dither re-quantize internally) and then push residual error to neighboring pixels in various patterns.

## General Tips
- Run a distance metric first if you need perceptual color selection, then feed the result into a dither.
- All dithers resize the source to 128×96 and emit `.s` files with `.Resolution` and `.PixelScreen` already configured.
- Combine with previews by exporting an intermediate PNG (edit the script) before generating assembly to see the pattern at a glance.

## Algorithms

### floyd-steinberg.py
Canonical diffusion kernel (7/16, 3/16, 5/16, 1/16) tuned to ARMLite stride.
```bash
python algorithms/dithers/floyd-steinberg.py glass.png -o glass_fs.s
```
Parameters: `image`, `-o/--output`.

### jarvis_judice_ninke.py
Spreads error to 12 neighbors for a smoother look on photographic scenes.
```bash
python algorithms/dithers/jarvis_judice_ninke.py sky.png -o sky_jjn.s
```
Parameters: `image`, `-o/--output`.

### stucki.py
Balanced diffusion that limits directional artifacts, great for geometric sprites.
```bash
python algorithms/dithers/stucki.py ship.png -o ship_stucki.s
```
Parameters: `image`, `-o/--output`.

### atkinson.py
Lightweight kernel that deliberately loses some error, producing a softer, retro aesthetic.
```bash
python algorithms/dithers/atkinson.py ui.png -o ui_atkinson.s
```
Parameters: `image`, `-o/--output`.

### sierra.py
Sierra family (3, 2, and Lite variants) with runtime selection.
```bash
python algorithms/dithers/sierra.py mosaic.png --variant sierra-lite -o mosaic_sierra.s
```
Parameters: `image`, `-o/--output`, `--variant` (`sierra3`, `sierra2`, `sierra-lite`, default `sierra3`).

Experiment with combining a perceptual quantizer (Wu, K-means, or SOM) followed by Sierra or JJN to get both accurate color placement and clean gradients.
