# DNLP
The goal of this project is to analyze the potential applications of Chronos, a framework for pretrained probabilistic time series models proposed by Amazon.
We introduce three main variations that efficiently work using Chronos in zero-shot prompting:
* ChronosStack: a stack learner that combines Chronos with LightGBM to include the possibility of handling covariates.
* ChronosTSClassifier: a time-series classification pipeline based on Chronos embeddings
* ChronosTSAD: a time-series anomaly detection pipeline

You can find the source code and paper from Amazon below:
* [paper](https://arxiv.org/pdf/2403.07815)
* [code](https://github.com/amazon-science/chronos-forecasting)
