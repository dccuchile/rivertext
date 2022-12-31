# IWEF: The Incremental Word Embedding Framework (This work is still in development)


Incremental Word Embedding Framework (IWEF) is an open-source library for modeling and
training different incremental word vector architectures proposed by the state-of-the-art.

It seeks to standardize many existing incremental word vector algorithms into a unified
framework to provide a standardized interface for:

* Encapsulating existing incremental word vector methods from previous work and designing new ones.
* Training text representation regarding text data streams.

IWEF also standardizes the training process through an interface similar to the `river` package. This standardization follows two training paradigms:

* Training one instance per time (`learn_one` method).
* Training a mini-batch of instances per time (`learn_many` method).

## Table Of Contents

The documentation follows the best practice for
project documentation as described by Daniele Procida
in the [Diátaxis documentation framework](https://diataxis.fr/)
and consists of four separate parts:

1. [Tutorials](tutorials.md)
2. [How-To Guides](how-to-guides.md)
3. [Reference](reference.md)
4. [Explanation](explanation.md)

Quickly find what you're looking for depending on
your use case by looking at the different pages.

## Team


* [Gabriel Iturra](https://giturra.github.io/)
* [Felipe Bravo-Marquez](https://felipebravom.com/)


## Acknowledgements
