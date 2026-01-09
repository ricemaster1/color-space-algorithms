# Environment Manager Comparison — pyenv vs conda/mamba (+ direnv)

This file summarizes common choices and shows quick commands and `.envrc` snippets to compare workflows.

## Quick summary

- `venv` + `pip` + `direnv`: Minimal, portable, uses standard library; combine with `direnv` for per-directory activation.
- `pyenv` + `pyenv-virtualenv`: Manage multiple Python versions, create named virtualenvs. Good for reproducible Python-only projects.
- `conda` / `mamba`: Manage Python and binary packages (compiled dependencies) in one tool. `mamba` is a fast alternative to `conda`.


## Running the Toolchain

 
1. Get the code

	 Option A (recommended):

	 ```bash
	 git clone https://github.com/ricemaster1/color-space-algorithms.git
	 cd color-space-algorithms/algorithms
	 ```

	 Option B: download the ZIP from the repo page and extract the `algorithms` folder.

2. Overview of environment choices

	 - `pyenv` (+ `pyenv-virtualenv`): best when you need to manage multiple Python versions and create named virtualenvs.
	 - `conda` / `mamba`: best when you want a single tool for Python versions and binary packages; `mamba` is a faster drop-in replacement for `conda`.
	 - `pip` + `venv` (stdlib) + `direnv`: lightweight, portable, and recommended when you prefer standard tooling.

	 Pick one workflow — mixing managers can produce confusing PATHs and surprising interpreters.

3. Detailed workflows and examples

	 A. pyenv + pyenv-virtualenv (macOS / Linux)

	 - Install pyenv and pyenv-virtualenv:

		 ```bash
		 # macOS
		 brew install pyenv pyenv-virtualenv

		 # Linux (example installer)
		 curl https://pyenv.run | bash
		 # then install pyenv-virtualenv per its README
		 ```

	 - Install a Python version and create a virtualenv:

		 ```bash
		 pyenv install 3.10.11
		 pyenv virtualenv 3.10.11 armlite-algos
		 pyenv local armlite-algos   # creates .python-version
		 ```

	 - Example `.envrc` (keep it explicit):

		 ```bash
		 # use the pyenv-managed interpreter directly
		 layout python ~/.pyenv/versions/armlite-algos/bin/python
		 ```

	 - Then run:

		 ```bash
		 direnv allow   # grants the directory permission to set the environment
		 pip install -r requirements.txt
		 ```

	 Notes:
	 - Use `pyenv which python` and `python --version` to confirm the active interpreter.
	 - Prefer explicit paths in `.envrc` when reproducibility matters.

	 B. Conda / Mamba (all platforms)

	 - Install Miniconda (or Anaconda). For speed, install `mamba` after conda:

		 ```bash
		 # after conda/miniconda is installed
		 conda create -n armlite-algos python=3.10 -y
		 conda activate armlite-algos
		 conda install -n base -c conda-forge mamba -y   # optional, for faster installs
		 ```

	 - Example `.envrc` to integrate with direnv:

		 ```bash
		 use conda armlite-algos
		 ```

	 - Installing dependencies with mamba (faster) or conda:

		 ```bash
		 mamba install --yes --file requirements.txt -c conda-forge   # if you have mamba
		 # or
		 conda install --yes --file requirements.txt -c conda-forge
		 ```

	 Notes:
	 - `conda` environments are self-contained; `conda env export` produces an `environment.yml` for reproducibility.
	 - Use `mamba` when installing many compiled dependencies.

	 C. stdlib `venv` + pip + direnv (portable, minimal)

	 - Create and activate a venv:

		 ```bash
		 python3 -m venv .venv
		 source .venv/bin/activate
		 pip install -U pip
		 pip install -r requirements.txt
		 ```

	 - Example `.envrc` that uses the local venv:

		 ```bash
		 # activate the repo-local venv automatically
		 source .venv/bin/activate
		 ```

	 - Or use direnv's `layout python` to create and manage a venv automatically:

		 ```bash
		 # .envrc
		 layout python python3
		 ```

4. direnv quick setup

	 - Install `direnv` (Homebrew or distro package manager) and add the hook to your shell config:

		 ```sh
		 # zsh
		 eval "$(direnv hook zsh)"
		 ```

	 - Create an `.envrc` in the `algorithms` folder (examples above) and run:

		 ```bash
		 direnv allow
		 ```

	 - To revoke: `direnv deny` or edit `.envrc` then re-run `direnv allow`.

5. Installing dependencies

	 - With pip (venv / pyenv-managed venv):

		 ```bash
		 pip install -r requirements.txt
		 ```

	 - With conda / mamba:

		 ```bash
		 mamba install --yes --file requirements.txt -c conda-forge
		 # or
		 conda install --yes --file requirements.txt -c conda-forge
		 ```

6. Verify and run

	 - Confirm interpreter and PATH:

		 ```bash
		 which python
		 python --version
		 which pip
		 pip --version
		 ```

	 - Run a quick smoke test (example):

		 ```bash
		 python -c "import sys; print(sys.version)"
		 python your_script.py --help
		 ```

Best practices and troubleshooting

- Don't mix pyenv global/local with system Python and conda in the same shell session unless you understand the PATH ordering.
- When using `direnv`, prefer explicit interpreter paths in `.envrc` for reproducibility across machines.
- If `conda activate` doesn't work, run `conda init` for your shell and restart the terminal.
- If `direnv` does not auto-activate, ensure your shell loads the direnv hook and that the `.envrc` is allowed.

If you'd like, I can commit these updates or run a quick local smoke test; tell me which you'd prefer next.
**Best Practices & Troubleshooting:**

- Always add `eval "$(direnv hook zsh)"` (or bash/fish) to your shell config.
- If `pyenv virtualenv-init` is not found, ensure pyenv-virtualenv is installed and your shell config sources its init script.
- If direnv is not auto-activating/deactivating, check that your shell is loading the direnv hook and that `.envrc` exists and is allowed.
- To avoid confusion, prefer explicit Python paths in `.envrc` when using pyenv-virtualenv.
- Use `pyenv which python` or `pyenv which pip` to confirm the Python interpreter being used.
- If you see the wrong Python version, check your shell's PATH and pyenv global/local settings.

For more, see [direnv.net](https://direnv.net/) and [pyenv wiki](https://github.com/pyenv/pyenv/wiki/Common-build-problems).

4. Install dependencies:
	```bash
	# pyenv or pip + direnv
	pip install -r requirements.txt

	# conda 
	conda install --file requirements.txt -c conda-forge
	```s

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
