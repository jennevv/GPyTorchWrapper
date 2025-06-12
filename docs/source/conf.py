import sys
from pathlib import Path
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GPyTorchWrapper'
copyright = '2025, Jenne Van Veerdeghem'
author = 'Jenne Van Veerdeghem'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0,str(Path('../..').resolve()))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
]

autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True
}

templates_path = ['_templates']
exclude_patterns = []

# options for typehints
always_document_param_types = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

