# About

RiverTex is an open-source library for modeling and training different incremental word vector architectures proposed by the state-of-the-art.

It seeks to standardize many existing incremental word vector algorithms into a unified framework to provide a standardized
interface and facilitate the development of new methods.

RiverTex provides two training paradigms:

* `learn_one`, which trains one instance at a time;

* and `learn_many`, which trains a mini-batch of instances at a time.

This allows for more efficient training of text representation models with text data streams.

RiverText also provides an interface similar to the [`river`](https://riverml.xyz/0.14.0/) package, making it easy for developers to use the library to quickly and easily train text representation models.

## Motivation and Goals

Incremental word embedding models are becoming increasingly important in NLP systems. They allow for the dynamic update of word embeddings as new data becomes available, avoiding the need for retraining a model from scratch.
Previous approaches to word embedding models have focused on batch training, which can become impractical when dealing large amounts of data. The development of incremental word embedding models has addressed this issue, offering a more efficient solution.

The main objectives we want to achieve with this library are:

* To provide a ready-to-use tool for updating word embeddings incrementally as new data becomes available.

* To offer a simple and flexible interface for integrating incremental word embeddings into NLP systems.

* To provide a set of utilities for developing and testing new incremental word embedding methods.

We aim to make this library a valuable resource for researchers and practitioners alike, providing a reliable solution for the dynamic update of word embeddings in NLP systems.


## Team

* [Gabriel Iturra](https://giturra.cl/)
* [Felipe Bravo-Marquez](https://felipebravom.com/)


## License

RiverText is licensed under the BSD 3-Clause License.

Details of the license on this [link](https://github.com/dccuchile/rivertext/blob/main/LICENSE).


## Contact

Please write to gabrieliturrab at ug.chile.cl for inquiries about the software. You are also welcome to do a pull request or publish an issue in the RiverText repository on Github.
