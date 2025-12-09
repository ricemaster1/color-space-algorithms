# ARMLite Algorithm Collection

This directory gathers every experimental quantizer, dither, distance metric, and color transform built for generating Peter Higginson's ARMLite sprites. Each script converts source imagery into ARMLite-compatible color names and assembly listings so you can mix and match pipelines to suit a given sprite.

## Feature Highlights
- Covers palette builders from classic Median Cut to neural approaches such as NeuQuant and SOM-based quantizers.
- Includes five error-diffusion dithers tuned for the 128 by 96 hi-res PixelScreen.
- Provides interchangeable color distance modules, including full Delta E families and Mahalanobis scoring.
- Supplies helper transforms for HSV, Lab, XYZ, and YCbCr so you can test alternate color theories before quantizing.
- Every script emits standard `.Resolution`, `.PixelScreen`, and per-pixel stores, making outputs drop-in ready for https://peterhigginson.co.uk/ARMlite/.

## Color Space Notes
Color accuracy on ARMLite depends on how you measure differences between RGB triples. Euclidean distance in RGB space favors midtones and tends to produce banding because RGB axes do not match perceptual response. CIE Lab separates luminance from chroma, enabling Delta E 76, 94, and 2000 calculations that better match human vision. Mahalanobis distance introduces covariance weighting, which is useful when sampling palettes from an image dominated by one hue family. HSV and HSL preserve hue angle so you can quantize without shifting color families, while XYZ is the bridge to Lab and YCbCr and is helpful when you want to bias luma driven dithers. These scripts let you chain transforms, distance metrics, and quantizers: convert to Lab or HSV, pick a perceptual distance, reduce the palette, then add a dither that matches the luminance structure you just established.

---

## Quantizers
Every quantizer emits a ready-to-run sprite by default. Unless noted, parameters follow the pattern below.

### `quantizer.py`
Baseline nearest neighbor mapper over the ARMLite palette.
- **Parameters**: `image` (input file), `-o/--output` (assembly destination, default `converted.s`).
- **Example**: `python algorithms/quantizers/quantizer.py poster.png -o poster.s`.

### `median_cut.py`
Histogram splitter that halves RGB space repeatedly.
- **Parameters**: `image`, `-o/--output`, `-c/--colors` (number of clusters, default 16).
- **Example**: `python algorithms/quantizers/median_cut.py sunset.png -c 48 -o sunset_mc.s`.

### `k_means.py`
Lab aware K-means clustering with random seeds and fixed iterations.
- **Parameters**: `image`, `-o/--output`. Palette size and distance settings are configured inside the script (edit constants if needed).
- **Example**: `python algorithms/quantizers/k_means.py city.png -o city_kmeans.s`.

### `wu_quantizer.py` and `wu_quantizer-001.py`
Variance minimization using Wu's algorithm. The `-001` build prints extra diagnostics.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/quantizers/wu_quantizer.py gradient.png -o gradient_wu.s`.

### `palette_graph_nn.py`
Constructs a palette graph and runs nearest neighbor passes to reduce adjacency error.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/quantizers/palette_graph_nn.py scan.png -o scan_graph.s`.

### `voronoi_palette.py`
Runs Lloyd relaxations in Lab space to distribute palette points evenly.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/quantizers/voronoi_palette.py neon.png -o neon_voronoi.s`.

### `kd_tree_palette.py`
KD-tree accelerated search for the closest ARMLite color.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/quantizers/kd_tree_palette.py flowers.png -o flowers_kd.s`.

### `bsp_partitioning.py`
Binary space partition builder with variance-based splitting.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/quantizers/bsp_partitioning.py night.png -o night_bsp.s`.

### `som_quantizer.py`
Self-organizing map that adapts palette entries across training epochs.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/quantizers/som_quantizer.py splash.png -o splash_som.s`.

### `neuquant.py`
Implementation of the NeuQuant neural quantizer used in GIF encoders.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/quantizers/neuquant.py sprite.png -o sprite_neuquant.s`.

### `octree.py`
Optimized octree palette builder with pruning for fast lookups.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/quantizers/octree.py rocks.png -o rocks_octree.s`.

### `node.py`
Node based quantizer that keeps a shortlist of candidate colors per pixel and lets neighbors vote.
- **Parameters**: `image`, `-o/--output`. After the CLI runs it prompts for `N`, the number of top colors to track (enter 1 through 10).
- **Example**: `python algorithms/quantizers/node.py portrait.png -o portrait_node.s`, then answer the prompt (for example `5`).

### `nthree.py`
Context aware quantizer that tests the best N candidates against orthogonal neighbors.
- **Parameters**: `image`, `-o/--output`. All tuning happens in code, so edit the constants if you need different behavior.
- **Example**: `python algorithms/quantizers/nthree.py logo.png -o logo_n3.s`.

All quantizers expect 128 by 96 output; provide any source image and the script resizes automatically.

## Dithers

### `floyd-steinberg.py`
Classic Floyd Steinberg diffusion with ARMLite stride baked in.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/dithers/floyd-steinberg.py glass.png -o glass_fs.s`.

### `jarvis_judice_ninke.py`
Twelve neighbor diffusion for ultra smooth gradients.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/dithers/jarvis_judice_ninke.py sky.png -o sky_jjn.s`.

### `stucki.py`
Balanced kernel that limits directional artifacts.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/dithers/stucki.py ship.png -o ship_stucki.s`.

### `atkinson.py`
Lightweight diffusion that intentionally drops some error for softer shading.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/dithers/atkinson.py ui.png -o ui_atkinson.s`.

### `sierra.py`
Implements Sierra3, Sierra2, and Sierra Lite.
- **Parameters**: `image`, `-o/--output`, `--variant` (choose `sierra3`, `sierra2`, or `sierra-lite`, default `sierra3`).
- **Example**: `python algorithms/dithers/sierra.py mosaic.png --variant sierra-lite -o mosaic_sierra.s`.

## Distance Metrics

These modules can be run independently to inspect how a distance choice maps the ARMLite palette, or imported by other converters.

### `distance_euclidean.py`
Simplest RGB squared error.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/distance_metrics/distance_euclidean.py ref.png -o ref_euclid.s`.

### `distance_cie76.py`
Uses Lab conversion plus the 1976 Delta E formula.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/distance_metrics/distance_cie76.py ref.png -o ref_76.s`.

### `distance_cie94.py`
Applies the 1994 Delta E formula with graphic arts constants.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/distance_metrics/distance_cie94.py ref.png -o ref_94.s`.

### `distance_ciede2000.py`
Full CIEDE2000 implementation.
- **Parameters**: `image`, `-o/--output`.
- **Example**: `python algorithms/distance_metrics/distance_ciede2000.py ref.png -o ref_2000.s`.

### `distance_delta_e.py`
Wrapper that lets you switch between Delta E families from the CLI.
- **Parameters**: `image`, `-o/--output`, `-m/--metric` (choose `ciede2000`, `cie94`, `cie76`, `2000`, `94`, `76`, or `hybrid`), `--cie94-application` (`graphic_arts` or `textiles`, used when the metric is `cie94` or `hybrid`).
- **Example**: `python algorithms/distance_metrics/distance_delta_e.py ref.png -m hybrid --cie94-application textiles -o ref_hybrid.s`.

### `distance_delta_e_neo.py`
Adaptive Delta E system that blends multiple formulas.
- **Parameters**: `image`, `-o/--output`, `-m/--metric` (default `adaptive`), `--cie94-application` (`graphic_arts` or `textiles`), `--chroma-threshold` (float, default 10.0), `--luminance-threshold` (float, default 12.0), `--hybrid-weights` (comma separated weights parsed by the script, default `0.6,0.3,0.1`).
- **Example**: `python algorithms/distance_metrics/distance_delta_e_neo.py ref.png -m adaptive --chroma-threshold 8 --luminance-threshold 15 --hybrid-weights 0.5,0.3,0.2 -o ref_neo.s`.

### `distance_mahalanobis.py`
Computes palette aware Mahalanobis distance with diagonal regularization.
- **Parameters**: `image`, `-o/--output`, `--epsilon` (float to stabilize covariance inversion, default 1e-3).
- **Example**: `python algorithms/distance_metrics/distance_mahalanobis.py ref.png --epsilon 0.0005 -o ref_maha.s`.

## Color Transforms

### `rgb_to_lab.py`
Converts to Lab and lets you reweight L, a, and b before downstream metrics.
- **Parameters**: `image`, `-o/--output`, `--weights` (comma separated weights for L,a,b, default `1,1,1`).
- **Example**: `python algorithms/color_transforms/rgb_to_lab.py forest.png --weights 1,1,0.2 -o forest_lab.png`.

### `rgb_to_hsv_hsl.py`
Outputs HSV or HSL representations.
- **Parameters**: `image`, `-o/--output`, `--space` (`hsv` or `hsl`, default `hsv`), `--weights` (comma separated weights applied to hue, saturation, and value or lightness).
- **Example**: `python algorithms/color_transforms/rgb_to_hsv_hsl.py art.png --space hsl --weights 1,0.5,0.5 -o art_hsl.png`.

### `rgb_to_xyz.py`
Transforms RGB to XYZ and lets you emphasize X, Y, or Z before exporting an assembly sprite.
- **Parameters**: `image`, `-o/--output`, `--weights` (comma separated weights for X,Y,Z, default `1,1,1`).
- **Example**: `python algorithms/color_transforms/rgb_to_xyz.py studio.png --weights 0.8,1.0,0.6 -o studio_xyz.png`.

### `rgb_to_ycbcr.py`
Generates Y, Cb, Cr channels for workflows that key off luminance.
- **Parameters**: `image`, `-o/--output`, `--weights` (comma separated weights for Y,Cb,Cr, default `1,1,1`).
- **Example**: `python algorithms/color_transforms/rgb_to_ycbcr.py film.png --weights 1.2,1,1 -o film_ycbcr.png`.

## Running the Toolchain
1. Create and activate a Python 3.10 environment using your preferred workflow:
	```bash
	# stdlib venv
	python3 -m venv .venv
	source .venv/bin/activate

	# pyenv + direnv
	pyenv virtualenv 3.10 armlite-algos
	echo "layout python python3" > .envrc
	direnv allow

	# conda
	conda create -n armlite-algos python=3.10
	conda activate armlite-algos
	```
2. Install dependencies: `pip install pillow webcolors numpy scipy` (some scripts mention extra packages in their headers; install those as needed).
3. Mix quantizers, distance metrics, color transforms, and dithers as needed. For example, convert to Lab, run `k_means.py` with `distance_ciede2000.py`, then apply `stucki.py` for fine diffusion.
4. Load any generated `.s` file into the ARMLite simulator, assemble, and press Run to view the sprite.

Pick any combination that suits your scene, and feel free to extend the collection with new modules that follow the same CLI conventions.
