RiverText (This work is still in development)
===================================================================================

RiverText is an open-source library for modeling and
training different incremental word vector architectures proposed by the state-of-the-art.

It seeks to standardize many existing incremental word vector algorithms into a unified
framework to provide a standardized interface for:

* Encapsulating existing incremental word vector methods from previous work and designing new ones.
* Training text representation regarding text data streams.

IWEF also standardizes the training process through an interface similar to the `river` package. This standardization follows two training paradigms:

* Training one instance per time (`learn_one` method).
* Training a mini-batch of instances per time (`learn_many` method).

The official documentation can be found at this [link](https://giturra.github.io/iwef/).

Installation
============

Requirements
------------

Contributing
------------

Development Requirements
------------------------

Testing
-------

Build the documentation
-----------------------

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
Please write to gabrieliturrab at ug.chile.cl for inquiries about the software. You are also welcome to do a pull request or publish an issue in the IWEF repository on Github.
