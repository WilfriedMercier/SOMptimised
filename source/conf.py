#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Hugo Plombat - LUPM <hugo.plombat@umontpellier.fr> & Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Configuration script for Sphinx documentation.
"""

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

###################################################
#               Project information               #
###################################################

project            = 'SOMptimised'
copyright          = '2022, Wilfried Mercier'
author             = 'Wilfried Mercier'
show_authors       = True

highlight_options  = {'default': {'lexers.python.PythonLexer'}}

extensions         = ['sphinx.ext.autodoc',
                      'sphinx.ext.mathjax',
                      'sphinx.ext.viewcode',
                      'sphinx.ext.autosummary',
                      'matplotlib.sphinxext.plot_directive',
                      'sphinx_execute_code',
                      'sphinx.ext.intersphinx',
                      'sphinx_copybutton',
                     ]

# The full version, including alpha/beta/rc tags
release            = '1.1'

# Add any paths that contain templates here, relative to this directory.
html_static_path   = ["_static"]
html_css_files     = ["custom.css"]

#######################################################
#               Options for HTML output               #
#######################################################

#html_theme         = "sphinxawesome_theme"
html_theme         = "sphinx_book_theme"
html_logo          = "../logo/logo.png"
html_title         = 'SOMptimised 1.1'

html_theme_options = {#'extra_header_links' : {
                      #"API": "/SOMptimised//API/index",
                      #"Tutorial": "/SOMptimised/tutorial/index"
                     #},
                     #"show_prev_next": True,
                     #"show_scrolltop": True,
                     "repository_url":'https://github.com/WilfriedMercier/SOMptimised',
                     "use_repository_button": True,
                     'use_issues_button':True,
                     'home_page_in_toc':True,
                     'show_navbar_depth':1
    }

#html_awesome_code_headers = True
#html_collapsible_definitions = True


rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight
  
.. role:: bash(code)
    :language: bash

.. _numpy: https://numpy.org/
.. _ndarray: https://numpy.org/doc/stable/reference/generated/numpy.array.html
.. _float64: https://numpy.org/doc/stable/user/basics.types.html
.. _pickle.dump: https://docs.python.org/3/library/pickle.html#pickle.dump
.. _pickle.load: https://docs.python.org/3/library/pickle.html#pickle.load
.. _Cigale: https://cigale.lam.fr/
.. _pandas: https://pandas.pydata.org/docs/index.html
.. _joblib: https://joblib.readthedocs.io/
.. _colorama: https://pypi.org/project/colorama/
"""

