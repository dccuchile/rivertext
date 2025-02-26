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
8. word-embeddings-benchmarks

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

Citation
========

Please cite the following paper if you use this package in an academic publication:

G. Iturra-Bocaz and F. Bravo-Marquez [RiverText: A Python Library for Training and Evaluating Incremental Word Embeddings from Text Data Stream. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2023)](https://dl.acm.org/doi/10.1145/3539618.3591908), Taipei, Taiwan.

```bibtex
@inproceedings{10.1145/3539618.3591908,
author = {Iturra-Bocaz, Gabriel and Bravo-Marquez, Felipe},
title = {RiverText: A Python Library for Training and Evaluating Incremental Word Embeddings from Text Data Streams},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3591908},
doi = {10.1145/3539618.3591908},
abstract = {Word embeddings have become essential components in various information retrieval and natural language processing tasks, such as ranking, document classification, and question answering. However, despite their widespread use, traditional word embedding models present a limitation in their static nature, which hampers their ability to adapt to the constantly evolving language patterns that emerge in sources such as social media and the web (e.g., new hashtags or brand names). To overcome this problem, incremental word embedding algorithms are introduced, capable of dynamically updating word representations in response to new language patterns and processing continuous data streams.This paper presents RiverText, a Python library for training and evaluating incremental word embeddings from text data streams. Our tool is a resource for the information retrieval and natural language processing communities that work with word embeddings in streaming scenarios, such as analyzing social media. The library implements different incremental word embedding techniques, such as Skip-gram, Continuous Bag of Words, and Word Context Matrix, in a standardized framework. In addition, it uses PyTorch as its backend for neural network training.We have implemented a module that adapts existing intrinsic static word embedding evaluation tasks for word similarity and word categorization to a streaming setting. Finally, we compare the implemented methods with different hyperparameter settings and discuss the results.Our open-source library is available at https://github.com/dccuchile/rivertext.},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {3027–3036},
numpages = {10},
keywords = {data streams, word embeddings, incremental learning},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
```

Team
====

- [Gabriel Iturra-Bocaz](https://giturra.cl/)
- [Felipe Bravo-Marquez](https://felipebravom.com/)

Contact
------------
Please write to gabriel.e.iturrabocaz@uis.no for inquiries about the software. You are also welcome to do a pull request or publish an issue in the RiverText repository on Github.
