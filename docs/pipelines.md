# Pipelines

The strength of this repository comes from mixing modules. Use the recipes below as starting points, then tweak parameters to match your sprite's subject matter.

## Pipeline Format (Coming Soon)
We are drafting a YAML schema that describes end-to-end jobs (transform → distance metric → quantizer → dither → exporter). Until then, treat the commands below as manual pipelines.

## Recipes

### Photographic Scene
1. Convert to Lab emphasizing luminance:
   ```bash
   python algorithms/color_transforms/rgb_to_lab.py photo.png --weights 1.2,1,1 -o photo_lab.png
   ```
2. Quantize with K-means using CIEDE2000 internally:
   ```bash
   python algorithms/quantizers/k_means.py photo_lab.png -o photo_lab_kmeans.s
   ```
3. Apply Jarvis-Judice-Ninke for smooth gradients:
   ```bash
   python algorithms/dithers/jarvis_judice_ninke.py photo.png -o photo_jjn.s
   ```

### Pixel Art / UI
1. Run Euclidean quantizer for sharp palette edges:
   ```bash
   python algorithms/quantizers/quantizer.py ui.png -o ui_quantized.s
   ```
2. Lightly dither with Atkinson to preserve crisp lines:
   ```bash
   python algorithms/dithers/atkinson.py ui.png -o ui_atkinson.s
   ```

### Neon / Sci-Fi
1. Convert to HSV to preserve hue transitions:
   ```bash
   python algorithms/color_transforms/rgb_to_hsv_hsl.py neon.png --space hsv -o neon_hsv.png
   ```
2. Voronoi quantization for even palette spacing:
   ```bash
   python algorithms/quantizers/voronoi_palette.py neon_hsv.png -o neon_voronoi.s
   ```
3. Sierra Lite dither for punchy highlights:
   ```bash
   python algorithms/dithers/sierra.py neon.png --variant sierra-lite -o neon_sierra.s
   ```

### Analytical / Wireframe
1. Score colors with Mahalanobis distance to respect dominant hues:
   ```bash
   python algorithms/distance_metrics/distance_mahalanobis.py wire.png --epsilon 0.0005 -o wire_maha.s
   ```
2. KD-tree quantizer for fast lookups:
   ```bash
   python algorithms/quantizers/kd_tree_palette.py wire.png -o wire_kd.s
   ```
3. Floyd-Steinberg for balanced diffusion:
   ```bash
   python algorithms/dithers/floyd-steinberg.py wire.png -o wire_fs.s
   ```

Contribute new recipes by adding sections here, preferably with before/after thumbnails that reference reproducible commands.
