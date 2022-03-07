# Project information

project = "enstat"
copyright = "2021, Tom de Geus"
author = "Tom de Geus"

# General configuration

autodoc_type_aliases = {"Iterable": "Iterable", "ArrayLike": "ArrayLike"}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for HTML output

html_theme = "sphinx_rtd_theme"
