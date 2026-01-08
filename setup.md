# Environment Manager Comparison — pyenv vs conda/mamba (+ direnv)

This file summarizes common choices and shows quick commands and `.envrc` snippets to compare workflows.

## Quick summary

- `pyenv` + `pyenv-virtualenv`: Manage multiple Python versions, create named virtualenvs. Good for reproducible Python-only projects.
- `conda` / `mamba`: Manage Python and binary packages (compiled dependencies) in one tool. `mamba` is a fast alternative to `conda`.
- `venv` + `pip` + `direnv`: Minimal, portable, uses standard library; combine with `direnv` for per-directory activation.

## Pros / Cons

- pyenv
  - Pros: precise Python-version control, integrates with system shells, small footprint.
  - Cons: you still manage virtualenvs (pyenv-virtualenv) or rely on venv; binary dependency installation uses pip (may compile).

- conda / mamba
  - Pros: manages Python versions and binary packages, environment export (`environment.yml`), cross-platform parity.
  - Cons: larger on disk, mixing conda and pyenv can be confusing.

- venv + pip + direnv
  - Pros: simple, uses stdlib tools, small and portable; `direnv` auto-activates per-directory.
  - Cons: pip may compile C extensions; reproducibility can be improved with pinned requirements.

## Quick commands

- pyenv + pyenv-virtualenv

```bash
pyenv install 3.10.11
pyenv virtualenv 3.10.11 armlite-algos
pyenv local armlite-algos
direnv allow
pip install -r requirements.txt
```

- conda / mamba

```bash
conda create -n armlite-algos python=3.10 -y
conda activate armlite-algos
mamba install --yes --file requirements.txt -c conda-forge
```

- venv + direnv

```bash
python3 -m venv .venv
echo "source .venv/bin/activate" > .envrc
direnv allow
pip install -r requirements.txt
```

## `.envrc` examples

- pyenv (explicit interpreter):

```bash
layout python ~/.pyenv/versions/armlite-algos/bin/python
```

- conda:

```bash
use conda armlite-algos
```

- venv:

```bash
source .venv/bin/activate
```

## Direnv + Conda (practical setup)

- Preferred (direnv captures and reverts cleanly):

```bash
source "$(direnv stdlib 2>/dev/null || true)"
use conda armlite-algos
```

- Alternative (explicit, works if stdlib helper isn't available):

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate armlite-algos
```

- Notes & troubleshooting:
  - Do not paste the full output of `direnv stdlib` into `.envrc`; source it as above.
  - If the env appears to "stick", run `conda deactivate` until no env is active, then let `direnv` manage activation.
  - Ensure the direnv shell hook is installed: add `eval "$(direnv hook bash)"` (or zsh) to your shell rc.

- Quick verification:

```bash
# with direnv active in the project
which python; python --version
echo "$CONDA_DEFAULT_ENV" "$CONDA_PREFIX"

# after leaving the project
which python; python --version
direnv status
direnv dump | head -c 400
```

## When to pick which

- Use `conda`/`mamba` when you rely on compiled dependencies from `conda-forge` or need a single tool to capture both Python and binary packages.
- Use `pyenv` when you need many Python versions and want small, reproducible per-version virtualenvs.
- Use `venv` + `direnv` when you want a minimal, standard toolchain that works everywhere and prefer pip-managed packages.

## Notes about mixing

- Avoid mixing `pyenv` and `conda` in the same shell session unless you know how PATH ordering will affect the active Python.
- When sharing the repo, include an `environment.yml` for conda users and `requirements.txt` for pip users to maximize portability.

---

If you'd like, I can add an `environment.yml` generated from a sample environment or a small script to create the recommended `.envrc` automatically.

## macOS (Apple Silicon) notes

- Quick checks (your output shows these typical Homebrew/Miniforge locations):

```bash
conda info --base    # e.g. /opt/homebrew/Caskroom/miniforge/base
which direnv         # e.g. /opt/homebrew/bin/direnv
which python         # ensure it points to the conda env when active
```

- Common pitfalls and fixes:
  - Use an arm64 Miniforge/Miniconda installer on Apple Silicon (Miniforge recommended).
  - Prefer installing `direnv` via Homebrew (`brew install direnv`) or in `conda` **base** (`mamba install -n base -c conda-forge direnv`) so the `direnv` binary is available to the login shell.
  - Ensure the direnv hook is in your `~/.zshrc`:

    ```bash
    echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
    ```

  - Run `conda init zsh` (once) or ensure `conda.sh` is sourced for non-interactive shells; the portable `.envrc` below uses `conda info --base` to find `conda.sh`.

- Portable `.envrc` (works across Miniforge/Miniconda/Anaconda paths):

```bash
# prefer direnv stdlib helper when present
source "$(direnv stdlib 2>/dev/null || true)"
use conda armlite-algos || {
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate armlite-algos
  else
    echo "warning: could not locate conda.sh; install conda/miniforge or direnv system-wide"
  fi
}
```

- Troubleshooting commands:

```bash
direnv status
direnv dump | head -n 80
conda env list
echo "$PATH"
```

- Notes:
  - The `(env)` prompt is cosmetic; rely on `which python` and `$CONDA_PREFIX` to confirm activation.
  - If activation appears inconsistent, `conda deactivate` until no env remains, then `cd` into the project to let `direnv` manage activation.
