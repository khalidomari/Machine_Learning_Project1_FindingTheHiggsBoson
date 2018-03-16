#Imports

# External Libraries
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *

#import implementations and necessary functions
import implementations as impl
import preprocessing as preproc
import random

def main():
	#Loading the Data
	# Training dataset
	DATA_TRAIN_PATH = '../data/train.csv' 
	y, X, ids = load_csv_data(DATA_TRAIN_PATH)
	# Testing Dataset
	DATA_TEST_PATH = '../data/test.csv' 
	y_t, X_t, ids_t = load_csv_data(DATA_TEST_PATH)

	#Separate training and testing sets into 4 different categories depending 
	#on the PRI_jet_num feature with index -8
	feature = -8
	X_cat = preproc.get_categories(X, feature=feature)
	X_t_cat = preproc.get_categories(X_t, feature=feature)

	#looop for every v in range 4 to obtain the 4 predictions, 
	#then concatenate and create submission file
	y_pred_all = []
	# Found using cross_validation

	# Setting best hyperparameters (the degree and the corresponding lambda) for each category
	degrees = [10, 10, 9, 9]
	lambdas = [0.00047508101621, 7.05480231072e-07, 0.000343046928631, 5.72236765935e-05]
	
	for v in range(4):
		# Extract category (test, train and labels)
	    Xv = X[X_cat[v]]
	    Xv_t = X_t[X_t_cat[v]]
	    y_v = y[X_cat[v]]

	    #Concatenante the train and testing set
	    all_Xv = np.concatenate((Xv, Xv_t), axis=0)

	    # find features (bad_features) with a unique value
	    bad_features = []
	    for i in range(len(all_Xv.T)):
	        if(len(np.unique(all_Xv.T[i])) == 1):
	            bad_features.append(i)

	    # Delete bad_features and fill missing values
	    all_Xv_c =  X_v = np.delete(all_Xv, bad_features, axis=1)
	    all_Xv_filled = preproc.fill_missing_values(all_Xv_c, tresh=1)

	    #Separate train and test
	    Xv_f = all_Xv_filled[:len(Xv)]
	    Xv_t_f = all_Xv_filled[len(Xv):]	    

	    #Standardize the dataset
	    tXv, mean_x, std_x = preproc.standardize(Xv_f)
	    tXv_t,  mean_x, std_x = preproc.standardize(Xv_t_f)

	    ### Generate model

	    final_degree = degrees[v]
	    best_lambda = lambdas[v]

	    # Build the polynomial basis, perform ridge regression
	    final_X = impl.build_poly(tXv, final_degree)
	    final_Xt = impl.build_poly(tXv_t, final_degree)

	    #Generate the model (Using Ridge Regression)
	    final_w, loss_ = impl.ridge_regression(y_v, final_X, best_lambda)

	    # Genereate prediction for this category
	    y_predv = predict_labels(final_w, final_Xt)
	    y_pred_all.append(y_predv)
	    p = len(X_cat[v])/len(X)

    ### Concatenate all predictions, and sort them by indices
	Xt_cat_all = [idx for sublist in X_t_cat for idx in sublist]
	y_pred = [yi for sublist in y_pred_all for yi in sublist]
	final_ypred = np.asarray(y_pred)[np.argsort(Xt_cat_all)]

	#Create Submission file
	OUTPUT_PATH = '../submissions/results__4categories_fillByCat_run.csv'

	create_csv_submission(ids_t, final_ypred, OUTPUT_PATH)
	print('Congratulations ........ Submission file created ::: ', OUTPUT_PATH)
	


if __name__ == "__main__":
	main()