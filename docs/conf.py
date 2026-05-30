# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, sys
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------
project   = "empirlab"
copyright = "2025, mofs0"
author    = "mofs0"
release   = "0.2.0"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # pull docstrings from code
    "sphinx.ext.napoleon",       # Google / NumPy style docstrings
    "sphinx.ext.viewcode",       # [source] links
    "sphinx.ext.autosummary",    # summary tables
    "sphinx.ext.intersphinx",    # cross-refs to NumPy, sklearn, etc.
    "sphinx_autodoc_typehints",  # render type hints
]

autosummary_generate = True
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "member-order":     "bysource",
}
napoleon_google_docstring  = False
napoleon_numpy_docstring   = True

intersphinx_mapping = {
    "python":  ("https://docs.python.org/3/", None),
    "numpy":   ("https://numpy.org/doc/stable/", None),
    "pandas":  ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ---------------------------------------------------
html_theme        = "furo"          # pip install furo
html_static_path  = ["_static"]
html_title        = "empirlab"
html_logo         = None
html_theme_options = {
    "sidebar_hide_name": False,
}

# -- autodoc tweaks ------------------------------------------------------------
add_module_names = False
