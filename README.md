IWEF: The Incremental Word Embedding  Framework
======================================================



*Word Embedding Fairness Evaluation* (WEFE) is an open source library for
measuring an mitigating bias in word embedding models.
It generalizes many existing fairness metrics into a unified framework and
provides a standard interface for:

- Encapsulating existing fairness metrics from previous work and designing
  new ones.
- Encapsulating the test words used by fairness metrics into standard
  objects called queries.
- Computing a fairness metric on a given pre-trained word embedding model
  using user-given queries.

WEFE also standardizes the process of mitigating bias through an interface similar
to the ``scikit-learn`` ``fit-transform``.
This standardization separates the mitigation process into two stages:

- The logic of calculating the transformation to be performed on the model (``fit``).
- The execution of the mitigation transformation on the model (``transform``).


The official documentation can be found at this `link <https://wefe.readthedocs.io/>`_.


Installation
============

There are two different ways to install WEFE:


To install the package with ``pip``  ::

    pip install wefe

- With conda:

To install the package with ``conda``::

    conda install -c pbadilla wefe


Requirements
------------

These package will be installed along with the package, in case these have not already been installed:

1. numpy
2. scipy
3. scikit-learn
4. scipy
5. pandas
6. gensim
7. plotly
8. requests
9. tqdm
10. semantic_version

Contributing
------------

You can download the code executing ::

    git clone https://github.com/dccuchile/wefe


To contribute, visit the `Contributing <https://wefe.readthedocs.io/en/latest/user_guide/contribute.html>`_ section in the documentation.

Development Requirements
------------------------

To install the necessary dependencies for the development, testing and compilation
of WEFE documentation, run ::

    pip install -r requirements-dev.txt


Testing
-------

All unit tests are in the wefe/tests folder. It uses ``pytest`` as a framework to
run them.

To run the test, execute::

    pytest tests

To check the coverage, run::

    pytest tests --cov-report xml:cov.xml --cov wefe

And then::

    coverage report -m


Build the documentation
-----------------------

The documentation is created using sphinx.
It can be found in the docs folder at the root of the project.
To compile the documentation, run:

.. code-block:: bash

    cd docs
    make html

Then, you can visit the documentation at ``docs/_build/html/index.html``

Changelog
=========

Version 0.4.0
-------------------
- 3 new bias mitigation methods (debias) implemented: Double Hard Debias, Half
  Sibling Regression and Repulsion Attraction Neutralization.
- The library documentation of the library has been restructured.
  Now, the documentation is divided into user guide and theoretical framework
  The user guide does not contain theoretical information.
  Instead, theoretical documentation can be found in the conceptual guides.
- Improved API documentation and examples. Added multilingual examples contributed
  by the community.
- The user guides are fully executable because they are now on notebooks.
- There was also an important improvement in the API documentation and in metrics and
  debias examples.
- Improved library testing mechanisms for metrics and debias methods.
- Fixed wrong repr of query. Now the sets are in the correct order.
- Implemented repr for WordEmbeddingModel.
- Testing CI moved from CircleCI to GithubActions.
- License changed to MIT.



Citation
=========


Please cite the following paper if using this package in an academic publication:

P. Badilla, F. Bravo-Marquez, and J. Pérez
`WEFE: The Word Embeddings Fairness Evaluation Framework In Proceedings of the
29th International Joint Conference on Artificial Intelligence and the 17th
Pacific Rim International Conference on Artificial Intelligence (IJCAI-PRICAI 2020), Yokohama, Japan. <https://www.ijcai.org/Proceedings/2020/60>`_

Bibtex:

.. code-block:: latex

    @InProceedings{wefe2020,
        title     = {WEFE: The Word Embeddings Fairness Evaluation Framework},
        author    = {Badilla, Pablo and Bravo-Marquez, Felipe and Pérez, Jorge},
        booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
                   Artificial Intelligence, {IJCAI-20}},
        publisher = {International Joint Conferences on Artificial Intelligence Organization},
        pages     = {430--436},
        year      = {2020},
        month     = {7},
        doi       = {10.24963/ijcai.2020/60},
        url       = {https://doi.org/10.24963/ijcai.2020/60},
        }


Team
====


- `Felipe Bravo-Marquez <https://felipebravom.com/>`_.

Contributors
------------


We thank all our contributors who have allowed WEFE to grow, especially
`stolenpyjak <https://github.com/stolenpyjak/>`_ and
`mspl13 <https://github.com/mspl13/>`_ for implementing new metrics.

We also thank `alan-cueva <https://github.com/alan-cueva/>`_ for initiating the development
of metrics for contextualized embedding models and
`harshvr15 <https://github.com/harshvr15/>`_ for the examples of multi-language bias measurement.

Thank you very much 😊!
