# Overview

## models

This module presents a unified implementation of various incremental word embedding methods, including the Word Context Matrix, Skipgram, and Continuous Bag of Words. The implementation incorporates these algorithms into a common interface, allowing for seamless integration and comparison of multiple models. Additionally, the module includes a base class to facilitate the addition of new incremental methods, further expanding the possibilities for exploration and development in the field.

* [IWVBase](./api/base.md)
* [IWordContextMatrix](./api/wcm.md)
* [IWord2Vec](./api/iw2v.md)
* iword2vec_utils:
    * [pytorch_nn](./api/pytorch_nn.md)
    * [UnigramTable](./api/unigram_table.md)
## evaluator

This module implements the Periodic Evaluation, a novel evaluation scheme designed to continually assess the quality of the incremental word embeddings generated by a model. The evaluation consists of a series of intrinsic NLP tasks, including analogies, similarities, and categorization, which are applied in intervals of "p" instances processed during the training process.

* [PeriodicEvaluation](./api/evaluator.md)

## utils

The shared utility classes and functions are a crucial aspect of the software architecture. This module comprises a vocabulary class and a stream tweets class, which are essential components for efficiently handling the data. The vocabulary class is responsible for storing the text vocabulary. In contrast, the stream tweets class reads the tweets from a text file, ensuring that the data is not stored entirely in memory.

* [TweetStream](./api/data.md)
* [Vocab](./api/vocab.md)