#imports
import implementations as impl
import numpy as np
from cross_validation import *
import random

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    # set 0s to 1 in order to avoid the division by zero problem
    std_x = [i if i != 0 else 1 for i in std_x]
    x = x / std_x
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x

def fill_missing_values(X_, deg=1, tresh = 1, lambda_=1e-7):
    # Create a dictionary to store the index of the feature with -999 value as key, and the corresponding indices as value
    X = X_.copy()
    unknown_dict = find_bad_features(X)
    
    # Get bad/good features indices
    bad_features = list(unknown_dict.keys())

    # select feature to fill depending on the treshold
    features_to_fill = [i for i in bad_features if ((len(unknown_dict[i])/len(X)) < tresh)]
    
    features_to_ignore = bad_features.copy()

    for i in features_to_fill:
        features_to_ignore.remove(i)

    clean_features = np.delete(np.arange(len(X.T)), bad_features)

    clean_X = X.T[clean_features]
    # Ignoring very bad features (>tresh)
    # fill missing values using least squares
    for i in features_to_fill:
        clean_idx = list(np.delete(np.arange(len(X)), unknown_dict[i]))
        tx = clean_X.T[clean_idx]
        ys = X.T[i][clean_idx]

        bad_idx_by_feature = unknown_dict[i]
        w, _ = impl.least_squares(ys, tx)
        y_bad = np.dot(clean_X.T[bad_idx_by_feature], w)

        # Predict missing values
        for idx in bad_idx_by_feature:
            X[idx][i] = y_bad[i]
    feat_to_conserve = np.delete(np.arange(len(X.T)), features_to_ignore)
    return X.T[feat_to_conserve].T


def find_bad_features(X):
	'''Finds features containing -999 value'''
	unknown_dict = {}
	for i in range(len(X.T)):
		x = X.T[i]
		idx_list = np.where(np.asarray(x)==-999)[0]
		if len(idx_list) != 0:
			unknown_dict[i] = idx_list
	return unknown_dict

def split_data(y, X, ratio=0.3):
    '''split dataset to training and testing set with he corresponding ratio'''
    np.random.seed(50)
    N_test = int(len(X)*ratio)
    idx_list = np.arange(len(X))
    np.random.shuffle(idx_list)
    test_idx = idx_list[:N_test]
    train_idx = idx_list[N_test:]
    return y[train_idx], X[train_idx], y[test_idx], X[test_idx]

def get_categories(X, V = np.arange(4), feature = -8):
    ''' Split the dataset into 4 different categories depending on the categorical feature PRI_jet_num'''
    categories = []
    for v in V:
        non_zeros_idx = np.nonzero(np.asarray(X.T[feature])-v)[0]
        categories.append(np.delete(np.arange(len(X)), non_zeros_idx))
    return categories

def fill_cat_missing(X, X_cat, X_t_cat, v):
    ''' prepare the dataset (categrie) to the fill_missing_values by deleting features with unique
    unique value to avoid the Singular matrix problem, and fill the missing values by calling fill_missing_values function '''
    Xv = delete_cat_feature(X)[X_cat[v]]
    X_tv = delete_cat_feature(X_t)[X_t_cat[v]]
    all_X = np.concatenate((Xv, X_tv), axis=0)
    all_X_filled = preproc.fill_missing_values(all_X, tresh=1)
    X_f = all_X_filled[:len(Xv)]
    X_t_f = all_X_filled[len(Xv):]
    return X_f, X_t_f