# Experimental Designed

## Data

Our experiment uses a dataset of unlabeled tweets to simulate a text stream of tweets. Twitter provides an excellent source of text streams, given its widespread use and real-time updates from its users. We draw a set of ten million tweets in English from the Edinburgh corpus. This dataset is a collection of tweets from different languages for academic purposes and was downloaded from November 2009 to February 2010 using the [Twitter API](https://developer.twitter.com/en/docs/twitter-api). We hypothesize that using this dataset of tweets as a text stream would allow us to evaluate the performance of incremental WE methods in a realistic scenario, given the nature of social media text and its dynamic and evolving nature.

The tweets can be found [here](https://drive.google.com/file/d/1HBBmxY1WLatKW7WZeTtMz9w4gwtKFLjA/view?usp=sharing).

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

The full tables of combinations are the following:

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

It is important to mention that the vocabulary size in all configurations was set to capture 1,000,000 words. Additionally, the period value, p, utilized in our experiments was set to 320,000 instances, with a batch size of 32. This period value was selected as it represents the point at which the evaluator was called after processing 320,000 tweets. These parameters were carefully selected to effectively analyze the performance of the different incremental word embedding models. Also, the different settings presented in the tables were running for each test dataset explained before.


## Experiment Replication

### Machine

The experiment was executed in a machine with the following characteristics:

* CPU: Intel Core i7-11700K
* RAM: 128 GB DDR4
* HDD1: 500GB Crucial P2 PCIe M.2 NVME
* HDD2: 2TB Kingston NV1 PCIe M.2 NVME
* GPU0: GeForce RTX 3090 24GB
* GPU1: GeForce RTX 3090 24GB

### Replication

To replicate the experiment, the following steps should be followed:

* Download the tweet dataset from the provided source.
* Define the models for each hyperparameter setting as specified in the tables.
* Define a test dataset, such as MEN, Mturk, or AP, and use it to evaluate the performance of the models.
* Load the tweet data into memory using a data loader.
* Define the deep learning model, such as ICBOW or another, and initialize it with the desired hyperparameters.
* Create a PeriodicEvaluator object to evaluate the model on the test dataset.
* Run the model on the tweet data using the PeriodicEvaluator object, and output the evaluation results to a JSON file.
* Repeat the process for all hyperparameter settings, architectures, and test datasets of interest.

It is important to follow the same pre-processing steps and data cleaning techniques as used in the original experiment to ensure the comparability of the results. Additionally, it is important to document all steps taken to ensure the replicability of the experiment, as well as any issues or challenges encountered during the replication process. Finally, a thorough analysis of the results and comparison with the original experiment should be performed to ensure the validity of the findings.
 However, any test dataset from the word embedding benchmark library should work.

For example, for the model ICBOW, the MEN dataset and the hyperparameter:

* ```emb_size=300```
* ```window_size=3```
* ```neg_samples_sum=3```

```python
# Import required libraries
from rivertext.utils import TweetStream
from torch.utils.data import DataLoader
from rivertext.models import IWord2Vec
from rivertext.evaluator import PeriodicEvaluator
from web.datasets.similarity import fetch_MEN
from web.evaluate import evaluate_similarity

# Load tweet dataset
ts = TweetStream("path/to/tweet/dataset.txt")
dataloader = DataLoader(ts, batch_size=32)

# Define ICBOW model with specified hyperparameters
icbow = IWord2Vec(vocab_size=1000000, unigram_table_size=100000000, window_size=3, neg_samples_sum=3, emb_size=300, device="cuda:1")

# Define PeriodicEvaluator object for evaluating on MEN dataset
pe = PeriodicEvaluator(ts, icbow, golden_dataset=fetch_MEN, eval_func=evaluate_similarity, path_output_file="path/to/outputfile.json")

# Run the model with a period of 320,000 iterations
pe.run(p=320000)
```

In these tables we share our results by model:

- [ICBOW](https://docs.google.com/spreadsheets/d/187Mdv8aZmD_56kzJJmYons-6TvVjrGfpR7ejbWmHN_8/edit?usp=sharing)
- [ISG](https://docs.google.com/spreadsheets/d/1KhnJLmmPVgJGj-UuH1kA9idEELF7EoXiF22AjqTQlJM/edit?usp=sharing)
- [IWCM](https://docs.google.com/spreadsheets/d/1IpSM6tmJ0JAr0ZubWsyNuuJOj9aIeeanlSJTvKb22bA/edit)


## Ranking

After obtaining the results of a series of experiments, it is important to create an overall ranking to determine the optimal hyperparameter configuration for a given model and test dataset. The following steps are typically followed:

1. Compute the mean value of each hyperparameter configuration and test dataset based on the results obtained from the time series analysis. This step involves aggregating the results obtained from multiple iterations of the model with different hyperparameter settings, and calculating the mean performance of the model on a given test dataset for each hyperparameter configuration.

1. For each test dataset that was evaluated, calculate the average mean value across all intrinsic tasks. This step involves aggregating the results of multiple intrinsic tasks for a given test dataset, and calculating the average performance of the model across all tasks.

3. Finally, order the obtained average values to create a ranking of hyperparameter configurations, with the lower position indicating the optimal configuration. This step involves sorting the average values obtained in the previous step in ascending order, and assigning a rank to each hyperparameter configuration based on its performance.

It is important to note that this process should be carried out in an objective and reproducible manner, and that the ranking obtained should be used as a guide for selecting the optimal hyperparameter configuration, rather than as an absolute measure of performance. Additionally, the evaluation metrics used should be carefully chosen based on the specific requirements of the task and the characteristics of the data being used.

Finally, we have this table comparing the performance of all models and configurations:

| Position | Model | Emb. size | Win. size | Num. N.S | Context size | Mean MEN | Mean Mturk | Mean AP | Overall mean |
|----------|-------|-----------|-----------|----------|--------------|----------|------------|---------|--------------|
| 1        | ICBOW | 100       | 3         | 6        | -            | 0.488    | 0.439      | 0.294   | 0.407        |
| 2        | ICBOW | 300       | 3         | 8        | -            | 0.507    | 0.428      | 0.284   | 0.406        |
| 3        | ICBOW | 300       | 3         | 6        | -            | 0.508    | 0.416      | 0.289   | 0.404        |
| 4        | ICBOW | 300       | 3         | 10       | -            | 0.505    | 0.419      | 0.284   | 0.403        |
| 5        | ICBOW | 100       | 3         | 10       | -            | 0.483    | 0.419      | 0.302   | 0.401        |
| 6        | ICBOW | 300       | 2         | 6        | -            | 0.483    | 0.423      | 0.29    | 0.399        |
| 7        | ICBOW | 200       | 3         | 10       | -            | 0.488    | 0.423      | 0.282   | 0.398        |
| 8        | ICBOW | 200       | 3         | 8        | -            | 0.483    | 0.411      | 0.291   | 0.395        |
| 9        | ISG   | 100       | 1         | 10       | -            | 0.442    | 0.416      | 0.32    | 0.393        |
| 10       | ICBOW | 200       | 3         | 6        | -            | 0.478    | 0.414      | 0.284   | 0.392        |
| 11       | ICBOW | 100       | 3         | 8        | -            | 0.489    | 0.389      | 0.298   | 0.392        |
| 12       | ICBOW | 300       | 2         | 8        | -            | 0.476    | 0.414      | 0.282   | 0.391        |
| 13       | ICBOW | 300       | 2         | 10       | -            | 0.468    | 0.414      | 0.286   | 0.389        |
| 14       | ISG   | 100       | 1         | 8        | -            | 0.44     | 0.4        | 0.321   | 0.387        |
| 15       | ISG   | 100       | 1         | 6        | -            | 0.443    | 0.393      | 0.312   | 0.383        |
| 16       | ICBOW | 100       | 2         | 6        | -            | 0.448    | 0.407      | 0.284   | 0.379        |
| 17       | ICBOW | 200       | 2         | 8        | -            | 0.448    | 0.412      | 0.277   | 0.379        |
| 18       | ICBOW | 100       | 2         | 8        | -            | 0.454    | 0.387      | 0.292   | 0.378        |
| 19       | ISG   | 100       | 2         | 10       | -            | 0.421    | 0.399      | 0.309   | 0.376        |
| 20       | ICBOW | 200       | 2         | 6        | -            | 0.456    | 0.392      | 0.278   | 0.375        |
| 21       | ISG   | 100       | 2         | 8        | -            | 0.423    | 0.392      | 0.311   | 0.375        |
| 22       | ICBOW | 200       | 2         | 10       | -            | 0.452    | 0.395      | 0.273   | 0.373        |
| 23       | ICBOW | 100       | 2         | 10       | -            | 0.44     | 0.388      | 0.284   | 0.371        |
| 24       | ISG   | 100       | 2         | 6        | -            | 0.427    | 0.381      | 0.303   | 0.37         |
| 25       | IWCM  | 100       | 3         | -        | 1000         | 0.44     | 0.343      | 0.319   | 0.367        |
| 26       | ICBOW | 300       | 1         | 6        | -            | 0.42     | 0.404      | 0.277   | 0.367        |
| 27       | IWCM  | 200       | 3         | -        | 1000         | 0.438    | 0.351      | 0.307   | 0.366        |
| 28       | IWCM  | 300       | 3         | -        | 1000         | 0.439    | 0.35       | 0.307   | 0.365        |
| 29       | ISG   | 100       | 3         | 8        | -            | 0.412    | 0.371      | 0.301   | 0.361        |
| 30       | IWCM  | 100       | 3         | -        | 750          | 0.429    | 0.336      | 0.318   | 0.361        |
| 31       | ICBOW | 300       | 1         | 10       | -            | 0.416    | 0.398      | 0.268   | 0.361        |
| 32       | IWCM  | 200       | 3         | -        | 750          | 0.428    | 0.339      | 0.311   | 0.359        |
| 33       | ISG   | 200       | 1         | 6        | -            | 0.413    | 0.392      | 0.272   | 0.359        |
| 34       | IWCM  | 100       | 2         | -        | 1000         | 0.417    | 0.34       | 0.318   | 0.358        |
| 35       | IWCM  | 200       | 2         | -        | 1000         | 0.419    | 0.343      | 0.308   | 0.357        |
| 36       | IWCM  | 300       | 3         | -        | 750          | 0.428    | 0.339      | 0.303   | 0.356        |
| 37       | ICBOW | 300       | 1         | 8        | -            | 0.423    | 0.37       | 0.276   | 0.356        |
| 38       | IWCM  | 100       | 3         | -        | 500          | 0.43     | 0.338      | 0.297   | 0.355        |
| 39       | IWCM  | 300       | 2         | -        | 1000         | 0.42     | 0.334      | 0.308   | 0.354        |
| 40       | IWCM  | 100       | 2         | -        | 750          | 0.404    | 0.34       | 0.319   | 0.354        |
| 41       | IWCM  | 200       | 3         | -        | 500          | 0.432    | 0.331      | 0.29    | 0.351        |
| 42       | ISG   | 200       | 1         | 10       | -            | 0.41     | 0.37       | 0.272   | 0.351        |
| 43       | ISG   | 100       | 3         | 10       | -            | 0.392    | 0.358      | 0.301   | 0.35         |
| 44       | IWCM  | 200       | 2         | -        | 750          | 0.403    | 0.339      | 0.309   | 0.35         |
| 45       | ISG   | 100       | 3         | 6        | -            | 0.406    | 0.345      | 0.3     | 0.35         |
| 46       | ISG   | 200       | 1         | 8        | -            | 0.408    | 0.372      | 0.266   | 0.348        |
| 47       | IWCM  | 300       | 3         | -        | 500          | 0.431    | 0.323      | 0.286   | 0.347        |
| 48       | IWCM  | 300       | 2         | -        | 750          | 0.404    | 0.335      | 0.3     | 0.347        |
| 49       | ISG   | 300       | 1         | 6        | -            | 0.407    | 0.364      | 0.268   | 0.346        |
| 50       | ISG   | 300       | 1         | 10       | -            | 0.395    | 0.381      | 0.263   | 0.346        |
| 51       | ICBOW | 200       | 1         | 8        | -            | 0.394    | 0.379      | 0.264   | 0.346        |
| 52       | IWCM  | 300       | 1         | -        | 1000         | 0.392    | 0.346      | 0.297   | 0.345        |
| 53       | IWCM  | 100       | 2         | -        | 500          | 0.4      | 0.341      | 0.294   | 0.345        |
| 54       | ISG   | 200       | 2         | 6        | -            | 0.397    | 0.368      | 0.266   | 0.343        |
| 55       | IWCM  | 200       | 1         | -        | 1000         | 0.387    | 0.341      | 0.302   | 0.343        |
| 56       | ISG   | 300       | 1         | 8        | -            | 0.398    | 0.369      | 0.257   | 0.342        |
| 57       | IWCM  | 200       | 2         | -        | 500          | 0.404    | 0.334      | 0.285   | 0.341        |
| 58       | IWCM  | 100       | 1         | -        | 1000         | 0.379    | 0.329      | 0.308   | 0.338        |
| 59       | IWCM  | 200       | 1         | -        | 750          | 0.36     | 0.372      | 0.283   | 0.338        |
| 60       | IWCM  | 300       | 1         | -        | 750          | 0.363    | 0.374      | 0.276   | 0.338        |
| 61       | ICBOW | 200       | 1         | 6        | -            | 0.401    | 0.344      | 0.265   | 0.337        |
| 62       | ICBOW | 100       | 1         | 6        | -            | 0.382    | 0.368      | 0.258   | 0.336        |
| 63       | IWCM  | 300       | 2         | -        | 500          | 0.403    | 0.323      | 0.281   | 0.336        |
| 64       | ISG   | 200       | 2         | 10       | -            | 0.395    | 0.346      | 0.261   | 0.334        |
| 65       | ISG   | 200       | 2         | 8        | -            | 0.398    | 0.341      | 0.263   | 0.334        |
| 66       | ICBOW | 200       | 1         | 10       | -            | 0.384    | 0.357      | 0.261   | 0.334        |
| 67       | ICBOW | 100       | 1         | 10       | -            | 0.372    | 0.357      | 0.261   | 0.33         |
| 68       | IWCM  | 100       | 1         | -        | 750          | 0.354    | 0.344      | 0.29    | 0.329        |
| 69       | ISG   | 200       | 3         | 8        | -            | 0.372    | 0.355      | 0.259   | 0.329        |
| 70       | ICBOW | 100       | 1         | 8        | -            | 0.378    | 0.35       | 0.257   | 0.328        |
| 71       | ISG   | 200       | 3         | 10       | -            | 0.383    | 0.345      | 0.256   | 0.328        |
| 72       | ISG   | 200       | 3         | 6        | -            | 0.385    | 0.34       | 0.256   | 0.327        |
| 73       | IWCM  | 200       | 1         | -        | 500          | 0.351    | 0.362      | 0.267   | 0.327        |
| 74       | ISG   | 300       | 2         | 8        | -            | 0.382    | 0.345      | 0.251   | 0.326        |
| 75       | IWCM  | 100       | 1         | -        | 500          | 0.344    | 0.347      | 0.285   | 0.325        |
| 76       | IWCM  | 300       | 1         | -        | 500          | 0.356    | 0.358      | 0.261   | 0.325        |
| 77       | ISG   | 300       | 2         | 6        | -            | 0.379    | 0.335      | 0.256   | 0.323        |
| 78       | ISG   | 300       | 2         | 10       | -            | 0.381    | 0.337      | 0.25    | 0.323        |
| 79       | ISG   | 300       | 3         | 6        | -            | 0.368    | 0.341      | 0.249   | 0.319        |
| 80       | ISG   | 300       | 3         | 10       | -            | 0.365    | 0.317      | 0.235   | 0.305        |
| 81       | ISG   | 300       | 3         | 8        | -            | 0.354    | 0.312      | 0.243   | 0.303        |

As can be seen from the results, the neural network models ICBOW and ISG demonstrate superior performance, on average, compared to the non-neural network IWCM model. It is noteworthy that the ICBOW models attain better results with larger embedding and window sizes. In contrast, the ISG models perform optimally with smaller embedding and window sizes. In the case of the IWCM model, the effect of embedding and window sizes on performance is unclear. However, a trend towards improved performance with larger context sizes is observable.

Considering these findings in the context of the chosen evaluation metrics and the intrinsic tasks involved is important. The results suggest that the neural network architecture of the ICBOW and ISG models may significantly impact the performance, particularly concerning capturing semantic relationships between words. Additionally, the varying optimal configurations for the ICBOW and ISG models highlight the need for thorough experimentation and analysis when selecting hyperparameters in these models. Further research may also consider exploring the underlying mechanisms and reasons for the observed performance differences between the models.
