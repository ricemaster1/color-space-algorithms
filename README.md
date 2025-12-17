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

## Documentation Hub
- `docs/quantizers.md` – palette builders, knobs, and pipeline pairings.
- `docs/dithers.md` – kernel diagrams plus variant guidance.
- `docs/distance-metrics.md` – perceptual math, CLI switches, and heatmaps.
- `docs/color-transforms.md` – RGB→Lab/HSV/XYZ/YCbCr helpers and weighting tricks.
- `docs/pipelines.md`, `docs/cli.md`, `docs/gallery.md` – pipelines, environment setup, and before/after templates.
- `docs/scripts/rgb_to_hsv_hsl.md` – script-level color theory, usage playbooks, and SVG assets (more single-script docs coming soon).

---

## Quantizers
Every quantizer shares the same core CLI (`image` plus optional `-o/--output`) and emits a ready-to-run sprite. Rather than maintain per-script tables here, jump to `docs/quantizers.md` for expanded coverage, including knob breakdowns, strengths, and recommended pairings.

Cheat sheet:
- **Baseline mappers** (`quantizer.py`, `kd_tree_palette.py`, `node.py`, `nthree.py`) provide quick palette-aligned exports.
- **Statistical splitters** (`median_cut.py`, `octree.py`, `bsp_partitioning.py`) excel on photographic material.
- **Perceptual relaxers** (`voronoi_palette.py`, `palette_graph_nn.py`, `som_quantizer.py`) maintain even Lab spacing.
- **Neural/heuristic** (`neuquant.py`, `wu_quantizer*.py`) mimic GIF encoders or expose diagnostics.

All quantizers resize inputs to 128×96 unless you override the constants noted in the docs. For usage examples, parameter matrices, and pipeline recipes (for example `lab → k_means → stucki`), see `docs/quantizers.md`.

## Dithers
Each diffusion script sticks to the same ergonomic CLI (`image`, optional `-o/--output`, plus `--variant` for Sierra). `docs/dithers.md` now hosts the weight diagrams, stride visualizations, and tuning charts so this README stays high level.

- **High fidelity**: `jarvis_judice_ninke.py`, `stucki.py`, and `sierra.py` (Sierra3/2/Lite) retain gradients with minimal artifacts.
- **Classic**: `floyd-steinberg.py` delivers the expected checkerboard texture.
- **Stylized**: `atkinson.py` intentionally drops error for softer shading.

Consult the docs page for before/after frames, kernel math, and when to downshift error diffusion to match a given quantizer.

## Distance Metrics
Use these modules as standalone inspectors or import them directly inside quantizers. The fast CLI reference—including Delta E flags, Mahalanobis regularizers, and the adaptive hybrid presets—lives in `docs/distance-metrics.md` along with palette heatmaps that visualize how each score warps color space.

- **Euclidean / RGB** (`distance_euclidean.py`) for quick tests where speed matters.
- **CIE families** (`distance_cie76.py`, `distance_cie94.py`, `distance_ciede2000.py`, `distance_delta_e.py`, `distance_delta_e_neo.py`) when you need perceptual accuracy or hybrid blends.
- **Mahalanobis** (`distance_mahalanobis.py`) if your palette samples are biased toward one hue range and you want covariance weighting.

Pick textile vs graphic arts constants, hybrid weights, and epsilon stabilizers using the tables in the docs page.

## Color Transforms
Each helper exposes the same ergonomics (`image`, optional `-o/--output`, and comma-separated weighting). Full conversion notes and weighting recipes now sit in `docs/color-transforms.md` so the README only needs the quick index below:

- `rgb_to_lab.py` – reweight luminance vs chroma ahead of Delta E comparisons.
- `rgb_to_hsv_hsl.py` – preserve hue angles when doing cel shading or node-based quantization.
- `rgb_to_xyz.py` – prep for Lab conversions or brightness-driven dithers.
- `rgb_to_ycbcr.py` – isolate luma if you plan on custom error kernels.

Head to the docs page for math references, output previews, and automation tips (for example, chaining the transforms in a Makefile target).

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
2. Install dependencies:
	```bash
	# pip (works in any environment)
	pip install -r requirements.txt

	# conda (reads from requirements.txt)
	conda install --file requirements.txt -c conda-forge
	```
3. Mix quantizers, distance metrics, color transforms, and dithers as needed. For example, convert to Lab, run `k_means.py` with `distance_ciede2000.py`, then apply `stucki.py` for fine diffusion.
4. Load any generated `.s` file into the ARMLite simulator, assemble, and press Run to view the sprite.

Pick any combination that suits your scene, and feel free to extend the collection with new modules that follow the same CLI conventions.
