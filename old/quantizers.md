# Quantizers

ARMLite quantizers reduce arbitrary RGB imagery to the simulator's fixed color names while preserving as much structure as possible. Every script resizes inputs to 128×96 by default, converts each pixel to the closest palette entry using the algorithm listed below, and emits a `.s` file ready for https://peterhigginson.co.uk/ARMlite/.

## General Workflow
1. Activate your Python environment and install dependencies (`pip install pillow webcolors numpy scipy`).
2. Pick a quantizer and run `python algorithms/quantizers/<script>.py input.png -o output.s`.
3. Optional: chain a distance metric (`distance_ciede2000.py`) or color transform (`rgb_to_lab.py`) before quantizing.
4. Load the `.s` file into ARMLite, assemble, and run.

## Algorithms

### quantizer.py
Baseline nearest neighbor lookup over the ARMLite palette.
```bash
python algorithms/quantizers/quantizer.py poster.png -o poster.s
```
Parameters: `image`, `-o/--output`.

### median_cut.py
Median cut histogram splitting with configurable palette size.
```bash
python algorithms/quantizers/median_cut.py sunset.png -c 48 -o sunset_mc.s
```
Parameters: `image`, `-o/--output`, `-c/--colors` (clusters, default 16).

### k_means.py
K-means clustering (Lab space) with deterministic iteration count and random seeding.
```bash
python algorithms/quantizers/k_means.py city.png -o city_kmeans.s
```
Parameters: `image`, `-o/--output`.

### wu_quantizer.py / wu_quantizer-001.py
Variance minimization using Wu's method; the `-001` variant prints diagnostics.
```bash
python algorithms/quantizers/wu_quantizer.py gradient.png -o gradient_wu.s
python algorithms/quantizers/wu_quantizer-001.py gradient.png -o gradient_wu_debug.s
```
Parameters: `image`, `-o/--output`.

### palette_graph_nn.py
Builds a palette graph and runs nearest-neighbor relaxation to minimize adjacency error.
```bash
python algorithms/quantizers/palette_graph_nn.py scan.png -o scan_graph.s
```
Parameters: `image`, `-o/--output`.

### voronoi_palette.py
Lloyd-relaxed Voronoi sampling in Lab space for evenly spaced palette points.
```bash
python algorithms/quantizers/voronoi_palette.py neon.png -o neon_voronoi.s
```
Parameters: `image`, `-o/--output`.

### kd_tree_palette.py
KD-tree accelerated nearest color search, ideal when you already have a palette JSON.
```bash
python algorithms/quantizers/kd_tree_palette.py flowers.png -o flowers_kd.s
```
Parameters: `image`, `-o/--output`.

### bsp_partitioning.py
Binary space partition palette builder that splits regions with the highest variance.
```bash
python algorithms/quantizers/bsp_partitioning.py night.png -o night_bsp.s
```
Parameters: `image`, `-o/--output`.

### som_quantizer.py
Self-organizing map that trains palette nodes across epochs.
```bash
python algorithms/quantizers/som_quantizer.py splash.png -o splash_som.s
```
Parameters: `image`, `-o/--output`.

### neuquant.py
Implementation of NeuQuant (a neural quantizer popularized by GIF encoders).
```bash
python algorithms/quantizers/neuquant.py sprite.png -o sprite_neuquant.s
```
Parameters: `image`, `-o/--output`.

### octree.py
Efficient octree palette generator with pruning and deduplication.
```bash
python algorithms/quantizers/octree.py rocks.png -o rocks_octree.s
```
Parameters: `image`, `-o/--output`.

### node.py
Node-based system that keeps a shortlist of candidate colors per pixel and lets neighbors vote. The CLI prompts for the number of candidates (1–10).
```bash
python algorithms/quantizers/node.py portrait.png -o portrait_node.s
# prompted: Enter number of closest colors to consider (e.g., 1-10): 5
```
Parameters: `image`, `-o/--output`, interactive `N` prompt.

### nthree.py
Context-aware top-N quantizer that evaluates candidates against orthogonal neighbors.
```bash
python algorithms/quantizers/nthree.py logo.png -o logo_n3.s
```
Parameters: `image`, `-o/--output`.

Feel free to add new quantizers under `algorithms/quantizers/` following the same CLI signature so they drop into existing pipelines.
