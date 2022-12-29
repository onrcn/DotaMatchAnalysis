# DOTA MATCH ANALYZER

## Quick Introduction:
This project:

* Scrapes the Dotabuff to build a dataset
* Uses various Machine Learning/Deep Learning technologies to predict the outcome of the given matches.

### Required Libraries:
- Tensorflow
- Pandas
- Scikit-Learn
- Numpy

## DISCLAIMER:
Tensorflow **doesn't** let you use the most recent version of python3. You can
change the python version from the beginning of my files.

```python3
#!/usr/bin/env python3.8
```

## Explaining the scripts:
### [classifiers][./classifiers.py]:
- The script splits the data and results datasets into training and
  testing sets using the `train_test_split function` from
  `sklearn.model_selection`.
- The script then imports the `RandomForestClassifier` class from
  `sklearn.ensemble` and creates an instance of this class called `rfc`. The `rfc`
  model is then fit to the training data using the `fit` method.
- The script uses the predict method of the rfc model to generate predictions
  for both the test data and the `test_data dataset`. The `accuracy_score`,
  function from `sklearn.metrics` is then used to calculate the accuracy of
  the model's predictions on both the test data and the `test_data` dataset.
- The script repeats this process for two other machine learning models: a
  decision tree classifier and a support vector classifier. The script imports
  the necessary classes, creates instances of the models, fits them to the
  training data, generates predictions, and calculates the accuracy of the
  predictions.

**_NOTE:_**  Datasets will be and can be updated by using the [fetch-script](./fetchers/fetch-script)
