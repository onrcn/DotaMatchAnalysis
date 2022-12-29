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
### [classifiers](./classifiers.py):
  * The first few lines import various libraries that are needed for the script, including os, tensorflow, pandas, and sklearn.
  * The script splits the data and results datasets into training and testing sets using the `train_test_split function` from `sklearn.model_selection`.
  * The script then imports the `RandomForestClassifier` class from `sklearn.ensemble` and creates an instance of this class called `rfc`. The `rfc` model is then fit to the training data using the `fit` method.
  * The script uses the predict method of the rfc model to generate predictions for both the test data and the `test_data dataset`. The `accuracy_score`, function from `sklearn.metrics` is then used to calculate the accuracy of the model's predictions on both the test data and the `test_data` dataset.
  * The script repeats this process for two other machine learning models: a decision tree classifier and a support vector classifier. The script imports the necessary classes, creates instances of the models, fits them to the training data, generates predictions, and calculates the accuracy of the predictions.

### [dense-neural-network](./dense-neural-network.py):
  * The first few lines import various libraries that are needed for the script, including os, tensorflow, pandas, and sklearn.
  * The script sets some environment variables related to TensorFlow, which is a library for machine learning.
  * The script loads a dataset from a CSV file using `pandas.read_csv` and processes the dataset by removing duplicates, dropping rows with missing values, and removing a column called 'Unnamed: 0'.
  * The script creates two variables, X and y, to store the feature data and target data from the dataset, respectively. The X variable stores the values of all columns except for the 'Result' column, and the y variable stores the values of the 'Result' column.
  * The script loads another dataset called `test_data` from a CSV file and processes it in a similar way as the first dataset. It also creates a `test_results` variable to store the values of the 'Result' column from the `test_data` dataset.
  * The script creates an instance of the MinMaxScaler class from sklearn.preprocessing and uses it to scale the values in the X variable.
  * The script then splits the scaled X and y datasets into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.
  * The script creates a neural network model using the Sequential class from tensorflow.keras. The model consists of several fully connected (Dense) layers with various activation functions and regularization techniques applied.
  * The script compiles the model using the compile method, specifying the optimizer, loss function, and metrics to be used during training.
  * The script creates an EarlyStopping callback object and passes it to the fit method of the model to interrupt training if the validation loss does not improve after a certain number of epochs.
  * The script trains the model on the training data using the fit method, specifying the number of epochs, batch size, and validation split to be used.
  * The script evaluates the model on the test data using the evaluate method and prints the test accuracy.
  * The script generates predictions for the test data and the `test_data dataset` using the predict method of the model.

### [five-fold-neural-network](./five-fold-neural-network.py)
  * The first few lines import various libraries that are needed for the script, including os, tensorflow, pandas, and sklearn.
  * The script sets some environment variables related to TensorFlow, which is a library for machine learning.
  * The script loads a dataset from a CSV file using `pandas.read_csv` and processes the dataset by removing duplicates, dropping rows with missing values, and removing a column called 'Unnamed: 0'.
  * The script creates two variables, X and y, to store the feature data and target data from the dataset, respectively. The X variable stores the values of all columns except for the 'Result' column, and the y variable stores the values of the 'Result' column.
  * The script loads another dataset called `test_data` from a CSV file and processes it in a similar way as the first dataset. It also creates a test_results variable to store the values of the 'Result' column from the `test_data` dataset.
  * The script creates an instance of the MinMaxScaler class from sklearn.preprocessing and uses it to scale the values in the X variable.
  * The script creates a KFold object from `sklearn.model_selection` with a specified number of splits and shuffle options.
  * The script enters a loop to iterate over the splits generated by the KFold object. On each iteration, the script splits the scaled X and y datasets into training and testing sets using the `train_index` and `test_index` variables.
  * The script creates neural network model using the Sequential class from tensorflow.keras. The model consists of several fully connected (Dense) layers with various activation functions and regularization techniques applied.
  * The script compiles the model using the compile method, specifying the optimizer, loss function, and metrics to be used during training.
  * The script creates an EarlyStopping callback object and passes it to the fit method of the model to interrupt training if the validation loss does not improve after a certain number of epochs.
  * The script trains the model on the training data using the fit method, specifying the number of epochs, batch size, and validation split to be used.
  * The script evaluates the model on the test data using the evaluate method and prints the test loss and accuracy.
  * The script generates predictions for the test data and the `test_data` dataset using the predict method of the model.
  * The script converts the predictions to binary values using a threshold of 0.5 and calculates the accuracy of the predictions on the `test_results` dataset using the accuracy_score function from sklearn.metrics.


**_NOTE:_**  Datasets will be and can be updated by using the [fetch-script](./fetchers/fetch-script)
