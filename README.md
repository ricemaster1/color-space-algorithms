# 🟦 Color Space Algorithms using ARMLite Simulator

Welcome to the **ARMLite Toolkit** — a modular Python suite for converting images into ARMLite-compatible sprite assembly code. The centerpiece is [`armlite.py`](armlite.py), a powerful CLI orchestrator for color transforms, quantization, dithering, and batch sprite workflows.

---

## 🎮 What is ARMLite?
ARMLite is a retro sprite engine and simulator. Try it live: [ARMLite Simulator](https://peterhigginson.co.uk/ARMlite)

---

## 🚀 Quickstart: Workflow Demo

```bash
# 1. Convert images to ARMLite sprite assembly
python armlite.py convert images/ -O output/ -a quantizer --preview

# 2. List available algorithms
python armlite.py convert -a help

# 3. Rename output files interactively or by pattern
python armlite.py rename output/ --mode pattern --pattern "sprite_{index}.s"
```

- See [`setup.md`](setup.md) for environment setup on Windows, macOS, and Linux.
- For full CLI options: `python armlite.py --help`

---

## 🗂️ Project Structure

- [`armlite.py`](armlite.py) — main CLI entry point and pipeline manager.
- [`lib/`](lib/) — shared color math, palette, and utility modules.
- [`lib/algorithms/`](lib/algorithms/) — all quantizers, dithers, distance metrics, and color transforms, organized by type.
 - [`lib/algorithms/color_transforms/`](lib/algorithms/color_transforms/)
 - [`lib/algorithms/distance_metrics/`](lib/algorithms/distance_metrics/)
 - [`lib/algorithms/dithers/`](lib/algorithms/dithers/)
 - [`lib/algorithms/quantizers/`](lib/algorithms/dithers/)

---

## 🧩 How It Works
- **Algorithm Discovery:** `armlite.py` auto-detects scripts in `lib/algorithms/` — add new quantizers, dithers, or transforms without changing the main script.
- **Pipeline Chaining:** Chain color transforms, quantization, and dithering via CLI arguments for custom sprite workflows.
- **Batch & Mapping:** Supports batch conversion, output mapping, preview generation, and interactive file selection.
- **Renaming:** Built-in CLI and GUI renaming for post-processing sprite outputs.

---

## 🎨 Color Spaces & Labels
- **Supported:** RGB, Lab, HSV, XYZ, YCbCr
- **Distance Metrics:** Euclidean, Delta E (76/94/2000), Mahalanobis
- **Quantizers:** Median Cut, Octree, KD-Tree, NeuQuant, SOM, Wu, Voronoi, Palette Graph NN
- **Dithers:** Floyd-Steinberg, Jarvis-Judice-Ninke, Stucki, Sierra, Atkinson

---

## 📚 Documentation

Full algorithm documentation is hosted on GitHub Pages:

**[📖 Browse the docs →](https://ricemaster1.github.io/color-space-algorithms/)**

Covers every quantizer, dither, distance metric, and color transform with usage instructions, CLI options, and implementation notes.

For environment setup, see [`setup.md`](setup.md).

---

## 🔗 External Resources
- [ARMLite Simulator](https://peterhigginson.co.uk/ARMlite)

---

> **Tip:** All outputs are compatible with the ARMLite Simulator. For advanced usage, see the docs and explore the modular algorithm folders.
