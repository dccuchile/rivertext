IWEF: The Incremental Word Embedding Framework (This work is still in development)
===================================================================================

Incremental Word Embedding Framework (IWEF) is an open-source library for modeling and
training different incremental word vector architectures proposed by the state-of-the-art.

It seeks to standardize many existing incremental word vector algorithms into a unified
framework to provide a standardized interface for:

* Encapsulating existing incremental word vector methods from previous work and designing new ones.
* Training text representation regarding text data streams.

IWEF also standardizes the training process through an interface similar to the `river` package. This standardization follows two training paradigms:

* Training one instance per time (`learn_one` method).
* Training a mini-batch of instances per time (`learn_many` method).

The official documentation can be found at this link.

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

Citation
========

Bibtex:

.. code-block:: latex

Team
====

- [Gabriel Iturra](https://github.com/giturra/)
- [Felipe Bravo-Marquez](https://felipebravom.com/)

Contributors
------------
