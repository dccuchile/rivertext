# Experimental Designed

## Data

Our experiment uses a dataset of unlabeled tweets to simulate a text stream of tweets. Twitter provides an excellent source of text streams, given its widespread use and real-time updates from its users. We draw a set of ten million tweets in English from the Edinburgh corpus. This dataset is a collection of tweets from different languages for academic purposes and was downloaded from November 2009 to February 2010 using the [Twitter API](https://developer.twitter.com/en/docs/twitter-api). We hypothesize that using this dataset of tweets as a text stream would allow us to evaluate the performance of incremental WE methods in a realistic scenario, given the nature of social media text and its dynamic and evolving nature.

## Experimental Setup

In our experimental investigation, we executed the Periodic Evaluation using a diverse range of datasets and hyperparameter settings. The evaluation was conducted on multiple architectural configurations ([Incremental Word Context Matrix](https://dccuchile.github.io/rivertext/api/wcm/) and [Incremental Word2Vec](https://dccuchile.github.io/rivertext/api/w2v/)) and intrinsic [test datasets](https://github.com/dccuchile/word-embeddings-benchmarks). The hyperparameters under consideration were the size of the embedding, the window size, the context size, and the number of negative samples. The results of this evaluation provide valuable insights into the performance of the different architectural configurations and hyperparameter settings, offering a comprehensive understanding of the subject under examination.

### Hyperparameters settings used

For the intrinsic test datasets, we used two datasets from the similarity tasks (MEN and Mturk) and one from the categorization task (AP).

The main hyperparameter configurations that we studied were:


* We evaluated the impact of three hyperparameters on neural network embedding:

    * Embedding size: refers to the dimensionality of the vector representation associated with each vocabulary word. Our configurations considered three different embedding sizes, including 100, 200, and 300.
    * Window size: This refers to the number of neighboring tokens used as the context for a target token. Our configurations utilized three different window sizes, including 1, 2, and 3.
    * The number of Negative samples: This refers to the number of negative instances that maximize the probability of a word being in the context of a target word. Our configurations considered three different numbers of negative samples, including 6, 8, and 10.

    Therefore, our experimental investigation considered a total of 27 configurations, comprising all combinations of the hyperparameters ($emb\_size \in {100, 200, 300}$, $window\_size \in {1, 2, 3}$, and $num\_ns \in {6, 8, 10}$) and for each of the architectural configurations and intrinsic test datasets.

* For the word context matrix embedding:
    
    * We leveraged the same configurations of the embedding size and window size as we did for the neural network embedding
    * Context size: represents the number of words associated with a vocabulary word based on the distributional hypothesis. The study involved three different context sizes, including 500, 750, and 1000.
    Therefore, 27 configurations were executed, incorporating all the possible combinations of ($emb\_size \in {100, 200, 300}$, $window\_size \in {1, 2, 3}$, and $context\_size \in {500, 750, 1000}$) for each intrinsic test dataset.


### Full Combinations of Hyperparameters

#### Incremental SkipGram and Incremental CBOW

| Emb_size | Window_size | ns_sample |
|----------|-------------|-----------|
| 100      | 1           | 6         |
| 100      | 1           | 8         |
| 100      | 1           | 10        |
| 100      | 2           | 6         |
| 100      | 2           | 8         |
| 100      | 2           | 10        |
| 100      | 3           | 6         |
| 100      | 3           | 8         |
| 100      | 3           | 10        |
| 200      | 1           | 6         |
| 200      | 1           | 8         |
| 200      | 1           | 10        |
| 200      | 2           | 6         |
| 200      | 2           | 8         |
| 200      | 2           | 10        |
| 200      | 3           | 6         |
| 200      | 3           | 8         |
| 200      | 3           | 10        |
| 300      | 1           | 6         |
| 300      | 1           | 8         |
| 300      | 1           | 10        |
| 300      | 2           | 6         |
| 300      | 2           | 8         |
| 300      | 2           | 10        |
| 300      | 3           | 6         |
| 300      | 3           | 8         |
| 300      | 3           | 10        |

#### Incremental Word Context Matrix

| Emb_size | Window_size | Context_size |
|----------|-------------|--------------|
| 100      | 1           | 500          |
| 100      | 1           | 750          |
| 100      | 1           | 1000         |
| 100      | 2           | 500          |
| 100      | 2           | 750          |
| 100      | 2           | 1000         |
| 100      | 3           | 500          |
| 100      | 3           | 750          |
| 100      | 3           | 1000         |
| 200      | 1           | 500          |
| 200      | 1           | 750          |
| 200      | 1           | 1000         |
| 200      | 2           | 500          |
| 200      | 2           | 750          |
| 200      | 2           | 1000         |
| 200      | 3           | 500          |
| 200      | 3           | 750          |
| 200      | 3           | 1000         |
| 300      | 1           | 500          |
| 300      | 1           | 750          |
| 300      | 1           | 1000         |
| 300      | 2           | 500          |
| 300      | 2           | 750          |
| 300      | 2           | 1000         |
| 300      | 3           | 500          |
| 300      | 3           | 750          |
| 300      | 3           | 1000         |

It is important to mention that the vocabulary size in all configurations was set to capture 1,000,000 words. Additionally, the period value, p, utilized in our experiments was set to 320,000 instances, with a batch size of 32. This period value was selected as it represents the point at which the evaluator was called after processing 320,000 tweets. These parameters were carefully selected in order to effectively analyze the performance of the different incremental word embedding models.   