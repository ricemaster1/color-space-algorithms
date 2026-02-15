# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = 'ARMLite Algorithm Suite'
copyright = '2026, ricemaster1'
author = 'ricemaster1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = 'index'
root_doc = 'index'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# MyST: enable dollar-sign math ($...$, $$...$$) and AMS environments
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

# MathJax 3 config (loaded automatically by sphinx.ext.mathjax)
mathjax3_config = {
    "tex": {
        "packages": {"[+]": ["ams"]},
    },
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2962ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#82b1ff",
        "color-brand-content": "#82b1ff",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}


def setup(app):
    app.add_css_file('custom.css')
