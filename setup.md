# Environment Setup Guide

This guide provides step-by-step instructions for setting up the environment on Windows, macOS, and Linux (major distributions only).

---

## Windows

**Step Zero (if necessary):**
- Install [Git](https://git-scm.com/download/win) (recommended).
- Optionally, use [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/) to install Git:
  ```powershell
  winget install --id Git.Git -e --source winget
  ```

**Step 1:** Clone the repository and change into it:
```powershell
git clone https://github.com/ricemaster1/color-space-algorithms.git
cd color-space-algorithms/algorithms
```

**Step 2:** Choose your environment manager and create/activate a venv:
- **venv (stdlib + pip):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **conda / mamba:**
  ```powershell
  conda create -n armlite-algos python=3.10 -y
  conda activate armlite-algos
  # optional: conda install -n base -c conda-forge mamba -y
  ```
- **pyenv-win:**
  - [pyenv-win](https://github.com/pyenv-win/pyenv-win) is a Windows port of pyenv, allowing you to manage multiple Python versions.
  - Install via pip or winget:
    ```powershell
    pip install pyenv-win --target %USERPROFILE%/.pyenv
    # or
    winget install pyenv-win
    ```
  - Usage:
    ```powershell
    pyenv versions
    pyenv install 3.10.11
    pyenv global 3.10.11
    ```
  - Limitations:
    - pyenv-win does not support virtualenvs natively; you must use `python -m venv` or another tool for environment isolation.
    - Some features from Unix pyenv (like plugins and shims) may not work identically on Windows.
    - Path management and shell integration are less seamless than on Unix systems.
    - For most users, conda or venv is simpler and more robust on Windows.

**Step 3:** (Optional) Install and configure [direnv](https://direnv.net/) (less common on Windows).

**Step 4:** Verify Python path and version:
```powershell
where python
python --version
```

**Step 5:** Install requirements:
```powershell
pip install -r requirements.txt
# or
conda install --yes --file requirements.txt -c conda-forge
```

**Step 6:** Finished.

---

## macOS

**Step Zero (if necessary):**
- Install [Homebrew](https://brew.sh/) (highly recommended):
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- Install Git if needed:
  ```bash
  brew install git
  ```

**Step 1:** Clone the repository and change into it:
```bash
git clone https://github.com/ricemaster1/color-space-algorithms.git
cd color-space-algorithms/algorithms
```

**Step 2:** Choose your environment manager and create/activate a venv:
- **venv (stdlib + pip):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **pyenv (+ pyenv-virtualenv):**
  ```bash
  brew install pyenv pyenv-virtualenv
  pyenv install 3.10.11
  pyenv virtualenv 3.10.11 armlite-algos
  pyenv local armlite-algos
  ```
- **conda / mamba (recommended):**
  ```bash
  conda create -n armlite-algos python=3.10 -y
  conda activate armlite-algos
  # optional: conda install -n base -c conda-forge mamba -y
  ```

**Step 3:** (Recommended) Install and configure [direnv](https://direnv.net/):
```bash
brew install direnv
# Add to your shell config (e.g., ~/.zshrc or ~/.bashrc):
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
```
- Example `.envrc` for conda (highest compatibility):
  ```bash
  # .envrc (dynamic, robust)
  ENV_NAME="armlite-algos"
  source "$(direnv stdlib 2>/dev/null || true)"
  if command -v use_conda >/dev/null 2>&1; then
    use conda "$ENV_NAME"
  else
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate "$ENV_NAME"
    else
      echo "[.envrc] Could not find conda.sh or use_conda; please check your conda installation."
    fi
  fi
  ```
- Example `.envrc` for venv:
  ```bash
  source .venv/bin/activate
  ```
- Troubleshooting: If you see `use_conda: command not found`, your direnv installation may lack the stdlib helper or your shell isn't loading it. The above dynamic example will fall back to manual activation if needed.

**Step 4:** Verify Python path and version:
```bash
which python
python --version
python -c "import sys; print(sys.executable)"
```

**Step 5:** Install requirements:
```bash
pip install -r requirements.txt
# or
mamba install --yes --file requirements.txt -c conda-forge
# or
conda install --yes --file requirements.txt -c conda-forge
```

**Step 6:** Finished.

---

## Linux (major distributions)

**Step Zero (if necessary):**
- Install Git if needed:
  ```bash
  sudo apt update && sudo apt install git   # Debian/Ubuntu
  sudo dnf install git                      # Fedora
  sudo pacman -S git                        # Arch
  ```

**Step 1:** Clone the repository and change into it:
```bash
git clone https://github.com/ricemaster1/color-space-algorithms.git
cd color-space-algorithms/algorithms
```

**Step 2:** Choose your environment manager and create/activate a venv:
- **venv (stdlib + pip):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **pyenv (+ pyenv-virtualenv):**
  ```bash
  curl https://pyenv.run | bash
  # Follow pyenv-virtualenv install instructions
  pyenv install 3.10.11
  pyenv virtualenv 3.10.11 armlite-algos
  pyenv local armlite-algos
  ```
- **conda / mamba (recommended):**
  ```bash
  conda create -n armlite-algos python=3.10 -y
  conda activate armlite-algos
  # optional: conda install -n base -c conda-forge mamba -y
  ```

**Step 3:** (Recommended) Install and configure [direnv](https://direnv.net/):
```bash
sudo apt install direnv   # Debian/Ubuntu
sudo dnf install direnv   # Fedora
sudo pacman -S direnv     # Arch
# Add to your shell config (e.g., ~/.bashrc or ~/.zshrc):
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
```
- Example `.envrc` for conda (highest compatibility):
  ```bash
  # .envrc (dynamic, robust)
  ENV_NAME="armlite-algos"
  source "$(direnv stdlib 2>/dev/null || true)"
  if command -v use_conda >/dev/null 2>&1; then
    use conda "$ENV_NAME"
  else
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate "$ENV_NAME"
    else
      echo "[.envrc] Could not find conda.sh or use_conda; please check your conda installation."
    fi
  fi
  ```
- Example `.envrc` for venv:
  ```bash
  source .venv/bin/activate
  ```
- Troubleshooting: If you see `use_conda: command not found`, your direnv installation may lack the stdlib helper or your shell isn't loading it. The above dynamic example will fall back to manual activation if needed.

**Step 4:** Verify Python path and version:
```bash
which python
python --version
python -c "import sys; print(sys.executable)"
```

**Step 5:** Install requirements:
```bash
pip install -r requirements.txt
# or
mamba install --yes --file requirements.txt -c conda-forge
# or
conda install --yes --file requirements.txt -c conda-forge
```

**Step 6:** Finished.

---

## Notes
- Pick one environment manager per session; mixing can cause confusing PATHs.
- For reproducibility, prefer conda/mamba or pyenv with explicit local settings.
- If using direnv, always run `direnv allow` after editing `.envrc`.
- If you skip Step Zero, continue from Step 1.
- For troubleshooting, see official docs:
  - [direnv](https://direnv.net/)
  - [pyenv](https://github.com/pyenv/pyenv)
  - [conda](https://docs.conda.io/)
  - [mamba](https://mamba.readthedocs.io/)
