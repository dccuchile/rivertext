RiverText
===================================================================================

RiverTex is an open-source library for modeling and training different incremental word vector architectures proposed by the state-of-the-art.

It seeks to standardize many existing incremental word vector algorithms into a unified framework to provide a standardized
interface and facilitate the development of new methods.

RiverTex provides two training paradigms:

* `learn_one`, which trains one instance at a time;

* and `learn_many`, which trains a mini-batch of instances at a time.

This allows for more efficient training of text representation models with text data streams.

RiverText also provides an interface similar to the [`river`](https://riverml.xyz) package, making it easy for developers to use the library to quickly
and easily train text representation models.

The official documentation can be found at this [link](https://dccuchile.github.io/rivertext/).

Installation
============

RiverText is meant to work with Python 3.10 and above. Installation can be done via ```pip```:

```
pip install rivertext
```

Requirements
------------

These package will be installed along with the package, in case these have not already been installed:

1. nltk
2. numpy
3. river
4. scikit_learn
5. scipy
6. torch
7. tqdm
8 word-embeddings-benchmarks

Contributing
------------

Development Requirements
------------------------

Testing
-------

All unit tests are in the rivertext/tests folder. It uses `pytest` as a framework to run them.

To run the test, execute:

```
pytest tests
```

To check the coverage, run:

```
pytest tests --cov-report xml:cov.xml --cov rivertext
```

And then:

```
coverage report -m
```

Build the documentation
-----------------------

The documentation is created using `mkdocs` and `mkdocs-material`. It can be found in the docs folder at the root of the project. First, you need to install:

```
pip install mkdocs
pip install "mkdocstrings[python]"
pip install mkdocs-material
```

Then, to compile the documentation, run:

```
mkdocs build
mkdocs serve
```

Changelog
=========

References
========

```bibtex
@article{iturra2023rivertext,
  title={A Python Library for Training and Evaluating Incremental Word Embeddings from Text Data Streams},
  author={Iturra-Bocaz, Gabriel and Bravo-Marquez, Felipe},
  year={2023}
}
```

Team
====

- [Gabriel Iturra-Bocaz](https://giturra.github.io/)
- [Felipe Bravo-Marquez](https://felipebravom.com/)

Contact
------------
Please write to gabrieliturrab at ug.chile.cl for inquiries about the software. You are also welcome to do a pull request or publish an issue in the RiverText repository on Github.
