# imports
import implementations as impl
import numpy as np
import matplotlib.pyplot as plt

def build_k_indices(N, k_fold=5, seed=50):
    """build k indices for k-fold."""
    size = int(N / k_fold)
    np.random.seed(seed)
    idx = np.random.permutation(N)
    return np.array_split(idx, k_fold)


def cross_validation(y, tx, mlfunction, split_number=5, lambda_=1e-6, gamma=0.001):
	'''Performs a ml_function given as parameters using cross validation on the training set split_number folds (5 as default value) '''

	# define empty lists to store train/test losses and accuracy
	train_loss_ 		= []
	test_loss_ 		= []
	train_accuracy_ 	= []
	test_accuracy_ 	= []

	# get k_indices
	k_indices = build_k_indices(len(y), split_number)

	for ki in range(len(k_indices)):

		# set the k'th indices as test, and others as training set
		#train_idx = np.asarray([k_indices[i] for i in np.delete( np.arange(len(k_indices)), ki)]).flatten()
		test_idx = np.asarray(k_indices[ki])
		train_idx = np.delete(np.arange(len(y)), test_idx)

		train_tX = tx[train_idx]
		train_y = y[train_idx]

		test_tX = tx[test_idx]
		test_y = y[test_idx]

		if(mlfunction == 'ridge_regression'):
			w, loss = impl.ridge_regression(train_y, train_tX, lambda_)
		elif(mlfunction == 'least_squares'):
			w, loss = impl.least_squares(train_y, train_tX)
		elif(mlfunction == 'logistic_regression'):
			w, loss = impl.logistic_regression(train_y, train_tX)
		elif(mlfunction == 'reg_logistic_regression'):
			w, loss = impl.reg_logistic_regression(train_y, train_tX, lambda_)

		elif(mlfunction == 'least_squares_sgd') :
			w, loss = impl.least_squares_SGD(train_y , train_tX, gamma)
		elif(mlfunction == 'least_squares_gd') : 
			w, loss = impl.least_squares_GD(train_y , train_tX, gamma)
		else:
			print('ERROR: ml_function not recognized')
			print('least_squares, least_squares_gd, least_squares_sgd, logistic_regression, reg_logistic_regression')
			return None


		# Calculate different losses and accuracy
		train_loss_.append(impl.compute_loss_mse(train_y, train_tX, w))
		test_loss_.append(impl.compute_loss_mse(test_y, test_tX, w))

		train_accuracy_ = impl.compute_accuracy(train_y, train_tX, w)
		test_accuracy_ = impl.compute_accuracy(test_y, test_tX, w)

	return np.mean(train_loss_), np.mean(test_loss_), np.mean(train_accuracy_), np.mean(test_accuracy_)


def plot_lambdas(lambdas, train_loss, test_loss, train_accuracy, test_accuracy, degree = 1 ,subplot = False):
	''' Plot mse_errors and prediction errors in function of lambdas.'''

	if(subplot): plt.subplot(211)
	fig1 = plt.figure()
	plt.semilogx(lambdas, train_loss, 'r.-', label='Train loss')
	plt.semilogx(lambdas, test_loss, 'b.-', label='Test loss')
	plt.legend()
	plt.title('Losses (degree = '+str(degree)+')')
	plt.xlabel('Lambda')
	plt.ylabel('Mean Square Error')

	if(subplot): plt.subplot(212)
	fig2 = plt.figure()
	plt.semilogx(lambdas, train_accuracy, 'r.-', label='Train accuracy')
	plt.semilogx(lambdas, test_accuracy, 'b.-', label='Test accuracy')
	plt.legend()
	plt.title('Accuracy (degree = '+str(degree)+')')
	plt.xlabel('Lambda')
	plt.ylabel('Accuracy')

	return fig1, fig2

def plot_degrees(degrees, train_loss, test_loss, train_accuracy, test_accuracy ,subplot = False):
	''' Plot mse_errors and prediction errors in function of degrees.'''

	if(subplot): plt.subplot(211)
	fig1 = plt.figure()
	plt.plot(degrees, train_loss, 'r.-', label='Train loss')
	plt.plot(degrees, test_loss, 'b.-', label='Test loss')
	plt.legend()
	plt.title('Losses')
	plt.xlabel('degree')
	plt.ylabel('Mean Square Error')

	if(subplot): plt.subplot(212)
	fig2 = plt.figure()
	plt.plot(degrees, train_accuracy, 'r.-', label='Train accuracy')
	plt.plot(degrees, test_accuracy, 'b.-', label='Test accuracy')
	plt.legend()
	plt.title('Accuracy)')
	plt.xlabel('degree')
	plt.ylabel('Accuracy')

	return fig1, fig2