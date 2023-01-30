RiverText (This work is still in development)
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

Contributing
------------

Development Requirements
------------------------

Testing
-------

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
@article{montiel2021river,
  title={River: machine learning for streaming data in Python},
  author={Montiel, Jacob and Halford, Max and Mastelini, Saulo Martiello and Bolmier, Geoffrey and Sourty,
    Raphael and Vaysse, Robin and Zouitine, Adil and Gomes, Heitor Murilo and Read, Jesse and Abdessalem,
    Talel and others},
  year={2021}
}

@article{bravo2022incremental,
  title={Incremental Word Vectors for Time-Evolving Sentiment Lexicon Induction},
  author={Bravo-Marquez, Felipe and Khanchandani, Arun and Pfahringer, Bernhard},
  journal={Cognitive Computation},
  volume={14},
  number={1},
  pages={425--441},
  year={2022},
  publisher={Springer}
}

@article{kaji2017incremental,
  title={Incremental skip-gram model with negative sampling},
  author={Kaji, Nobuhiro and Kobayashi, Hayato},
  journal={arXiv preprint arXiv:1704.03956},
  year={2017}
}
```

Team
====

- [Gabriel Iturra](https://giturra.github.io/)
- [Felipe Bravo-Marquez](https://felipebravom.com/)

Contact
------------
Please write to gabrieliturrab at ug.chile.cl for inquiries about the software. You are also welcome to do a pull request or publish an issue in the RiverText repository on Github.
