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

## 📚 Documentation
- [Quantizers](docs/quantizers.md)
- [Dithers](docs/dithers.md)
- [Distance Metrics](docs/distance-metrics.md)
- [Color Transforms](docs/color-transforms.md)
- [Setup Guide](setup.md)

---

> **Tip:** All outputs are compatible with the ARMLite Simulator. For advanced usage, see the docs and explore the modular algorithm folders.

```{toctree}
:maxdepth: 4
:glob:

algorithms/lib/algorithms/*/docs/*.md
