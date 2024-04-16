import os
import sys

sys.path.insert(0, os.path.abspath('../'))

project = 'galmoss'
copyright = '2023, Mi Chen'
author = 'Mi Chen'
release = '2.0.7'

extensions = ['sphinx.ext.autosummary', 'sphinx.ext.mathjax', 'sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.intersphinx', 'sphinx.ext.todo', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
autodoc_mock_imports = ["torch"]
html_static_path = ['_static']
