# Machine Learning Project 1

## Description

### `run.py`

This script generates the best predictions submited on Kaggle:
- Preprocess the data:
- Generates models
- Generates and save predictions

### `implementation.py`

This script contains the required machine learning algorithms for this project:
- Least Squares (normal equations, GD, SGD)
- Ridge Regression
- Logistic Regression
- Regularized Logistic Regression
As well as some helper functions for the machine learning algorithms compute_sigmoid and so on ...

### `preprocessing.py`

Contains all the functions to clean, split, standardize and predict the missing values in the datasets

### `cross_validation.py`

Contains the cross_validation function to perform cross validation on the training set in order to find the best hyperparameters and compare models.


### `proj1_helpers.py`

Contains helpers functions to load the dataset, predict labels and create submission as csv file.


## Generate predictions

Make sure the train.csv and test.csv (should be downloaded from https://www.kaggle.com/c/epfml-higgs/data) are in the data folder, and all py files are in scripts folder, then run 'run.py' by executing the command:
    `python run.py`
