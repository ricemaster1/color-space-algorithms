# Color Space Algorithms — ARMLite Toolkit

A modular Python toolkit that converts images into ARMLite-compatible sprite
assembly (`.s`) files. It chains color-space transforms, palette quantization,
and error-diffusion dithering through a single CLI entry point.

> **ARMLite Simulator** — load the generated `.s` files directly:
> <https://peterhigginson.co.uk/ARMlite>

---

## Quick Start

```bash
# Set up the environment (see setup.md for full OS-specific instructions)
pip install -r requirements.txt

# Convert a directory of images to ARMLite assembly
python armlite.py convert images/ -O output/ -a median_cut --preview

# List every discovered algorithm
python armlite.py convert -a help

# Batch-rename outputs with a pattern
python armlite.py rename output/ --mode pattern --pattern "sprite_{index}.s"
```

See [setup.md](setup.md) for environment setup on **Windows**, **macOS**, and **Linux** (conda, pyenv, direnv).

---

## CLI Reference

### `convert`

```
python armlite.py convert [input] [options]
```

| Flag | Description |
|---|---|
| `-O`, `--output-dir` | Directory for `.s` output (default: `output`) |
| `-a`, `--algo` | Algorithm slug or `help` to list |
| `-n` | Top-N palette size (default: 3) |
| `--preview` | Save preview PNGs alongside assembly |
| `--dry-run` | Print planned actions without writing |
| `--select-files` | Interactively pick input files |
| `--map-use` | Route outputs via keyword→folder mapping |
| `--progress` | Progress bar (`auto` / `on` / `off`) |

### `rename`

```
python armlite.py rename [directory] [options]
```

| Flag | Description |
|---|---|
| `--mode` | `cli` (per-file prompt), `pattern`, or `gui` |
| `--pattern` | Template with `{stem}`, `{index}`, `{ext}` |
| `--start` | Start index for pattern mode (default: 1) |
| `--dry-run` | Preview renames without applying |
| `--undo` | Undo the most recent rename batch |

---

## Project Structure

```
armlite.py                  CLI entry point & pipeline manager
lib/
├── palette.py              Palette extraction helpers
├── truecolor.py            24-bit color utilities
├── rename_utils.py         CLI / GUI / pattern rename engine
├── weight_tuner_gui.py     Interactive weight tuner
└── algorithms/
    ├── color_transforms/   RGB ↔ HSV/HSL, Lab, XYZ, YCbCr
    ├── distance_metrics/   Euclidean, CIE76, CIE94, CIEDE2000, ΔE, Mahalanobis
    ├── dithers/            Floyd-Steinberg, Atkinson, JJN, Stucki, Sierra
    └── quantizers/         Median Cut, Octree, KD-Tree, NeuQuant, SOM,
                            Wu, Voronoi, BSP, Palette Graph NN
```

Algorithm discovery is automatic — drop a new script into any category's `src/`
folder and `armlite.py` picks it up at runtime.

---

## Algorithms

### Color Transforms
RGB ↔ HSV / HSL · RGB ↔ CIE-Lab · RGB ↔ XYZ · RGB ↔ YCbCr

### Distance Metrics
Euclidean · CIE76 · CIE94 · CIEDE2000 · Delta E · Delta E Neo · Mahalanobis

### Quantizers
Median Cut · Octree · KD-Tree Palette · NeuQuant · SOM ·
Wu · Voronoi Palette · BSP Partitioning · K-Means · Palette Graph NN

### Dithers
Floyd-Steinberg · Jarvis-Judice-Ninke · Stucki · Sierra · Atkinson

---

## Documentation

Full algorithm docs (with LaTeX math, diagrams, and implementation notes) are
published to **GitHub Pages**:

**[Browse the docs →](https://ricemaster1.github.io/color-space-algorithms/)**

---

## Requirements

- Python ≥ 3.10
- Pillow, NumPy, SciPy, Matplotlib, webcolors, spectra

Install everything:

```bash
pip install -r requirements.txt
```

---

## License

See the repository for license details.
