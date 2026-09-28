# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cunumpy"
copyright = "2025, Max"
author = "Max"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "myst_parser",  # enable Markdown support
]

exclude_patterns = []

# Recognize both .rst and .md
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Activate the theme.
html_theme = "sphinx_book_theme"


html_static_path = ["_static"]
templates_path = ["_templates"]


# html_sidebars = {"**": []}

html_theme_options = {
    "repository_branch": "devel",
    "show_toc_level": 3,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/max-models/cunumpy",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
}
