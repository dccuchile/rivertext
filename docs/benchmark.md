# Benchmark

## Experiment replication

To ensure the successful replication of our experiment, we highly recommend following the steps outlined in this [guide](https://github.com/dccuchile/rivertext/tree/main/experiments). These steps are designed to provide a clear and concise set of instructions for reproducing our results and to ensure that the process is as straightforward as possible. By following these steps, you can obtain the necessary data and tools, run the required models and evaluations, and obtain an overall ranking of hyperparameter configurations that can guide future research.

It is important to note that some steps may require additional configuration or customization depending on the specific requirements of your research. Additionally, the performance of the models may be affected by factors such as the hardware used, the amount and quality of the training data, and the specific hyperparameters are chosen. Therefore, it is recommended that you carefully review the instructions and adapt them as necessary to suit your specific research needs.

By following these steps, you can conduct a reproducible and rigorous analysis of our model and obtain results that can inform future research in this field.

## Overall Ranking

In this ranking, we present the results of our investigation, which considers the performance of various models, hyperparameter settings, and intrinsic tasks. By ranking the results based on the mean values obtained for each configuration and test dataset, we provide an overview of the optimal hyperparameter settings for each model and the intrinsic task evaluated.

It is important to note that the ranking is based on a thorough analysis of the data obtained from our time series analysis and that the results may be influenced by various factors, such as the specific test datasets used, the quality and quantity of the training data, and the specific hardware and software used for the analysis. Nevertheless, our ranking provides a valuable guide for researchers in this field. It can be used to inform future research into the performance of word embedding models on various intrinsic tasks.

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
