# About

Incremental Word Embedding Framework (IWEF) is an open-source library for modeling and
training different incremental word vector architectures proposed by the state-of-the-art.

It seeks to standardize many existing incremental word vector algorithms into a unified
framework to provide a standardized interface for:

* Encapsulating existing incremental word vector methods from previous work and designing new ones.
* Training text representation regarding text data streams.

IWEF also standardizes the training process through an interface similar to the `river` package. This standardization follows two training paradigms:

* Training one instance per time (`learn_one` method).
* Training a mini-batch of instances per time (`learn_many` method).

The official documentation can be found at this [link](https://giturra.github.io/iwef/).


## Motivation and Goals

## Roadmap

## Team


* [Gabriel Iturra](https://giturra.github.io/)
* [Felipe Bravo-Marquez](https://felipebravom.com/)


## License

IWEF is licensed under the BSD 3-Clause License.

Details of the license on this [link](sasa).


## Contact

Please write to gabrieliturrab at ug.chile.cl for inquiries about the software. You are also welcome to do a pull request or publish an issue in the IWEF repository on Github.