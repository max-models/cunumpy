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
    "sphinx.ext.intersphinx",
    "numpydoc",  # NumPy-format docstrings (see contributing.md)
    "myst_parser",  # enable Markdown support
]

# The API reference (reference/*.md) is generated from the docstrings.
autodoc_typehints = "none"  # types are written in the docstrings
autodoc_member_order = "bysource"
nitpicky = True  # with -W in CI, a broken cross-reference fails the build
numpydoc_show_class_members = False
numpydoc_xref_param_type = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


def _defining_comment(name, obj):
    """The ``#:`` comment of a constant that a public module re-exports."""
    import sys

    from sphinx.pycode import ModuleAnalyzer

    for modname, module in list(sys.modules.items()):
        if not modname.startswith("cunumpy._") or getattr(module, name, None) is not obj:
            continue
        try:
            docs = ModuleAnalyzer.for_module(modname).find_attr_docs()
        except Exception:
            continue
        if ("", name) in docs:
            return list(docs[("", name)])
    return None


def _reexported_constant_doc(app, what, name, obj, options, lines):
    # constants imported from a private module are listed with ``autodata`` on
    # the reference pages; their ``#:`` comment is in the defining module
    if what == "data":
        comment = _defining_comment(name.rsplit(".", 1)[-1], obj)
        if comment:
            lines[:] = comment


def setup(app):
    app.connect("autodoc-process-docstring", _reexported_constant_doc)


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
