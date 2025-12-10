# CLI & Environment Guide

Most scripts share a common interface: positional `image` argument plus optional `-o/--output` for the destination `.s` file. A few add their own switches (documented on the category pages). This guide shows how to prepare your environment, run commands consistently, and automate batches.

## Environment Setup
```bash
# stdlib venv
python3 -m venv .venv
source .venv/bin/activate

# pyenv + direnv
pyenv virtualenv 3.10 armlite-algos
printf "layout python python3" > .envrc
Direnv allow

# conda
conda create -n armlite-algos python=3.10
conda activate armlite-algos

pip install pillow webcolors numpy scipy
```
(Install any extra dependencies mentioned in individual scripts.)

## Common Patterns
- **Single run**: `python algorithms/quantizers/median_cut.py input.png -c 32 -o out.s`
- **Preview first**: Duplicate the script and make it emit PNGs in addition to `.s` to check results without the simulator.
- **Batch processing**:
  ```bash
  for img in assets/*.png; do
      base=$(basename "$img" .png)
      python algorithms/quantizers/wu_quantizer.py "$img" -o builds/${base}_wu.s
  done
  ```
- **Chaining**: Use temporary files or pipes (where supported) to chain transforms → distance metrics → quantizers → dithers.

## Automation Ideas
- Add `make` targets (e.g., `make neon`) that encode end-to-end pipelines.
- Use `tox` or `nox` sessions to benchmark multiple algorithms automatically.
- Integrate with VS Code tasks so hitting a keybinding re-runs your favorite pipeline.

Stay consistent with paths by running commands from the repo root. All docs assume relative paths such as `algorithms/quantizers/...`.
