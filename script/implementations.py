# imports
import numpy as np
from proj1_helpers import *

##################################################################################
################################  Least Square GD  ###############################
##################################################################################
def least_squares_GD(y, tx, initial_w, max_iters=1000, gamma=1e-2):
    
    """
    Linear regression of mean-square error using gradient descent
    Args:
        y:  Target values 
        tx: Training data 
        initial_w:  initial w for the gradient descent
        max_iters:  maximum number of iteration, default_value=1000
        gamma:  step size
    Returns:
        ws[best]:  Optimal weight vector
        losses[best]: MSE error of ws[best]
    """
    current_w = initial_w
    loss = compute_loss_mse(y, tx, current_w)

    #Store the initial_w with the corresponding loss
    ws = [current_w]
    losses = [loss]

    for n_iter in range(max_iters): 
        gradient = compute_gradient_mse(y,tx,current_w)

        #Update w
        next_w = current_w - gamma*gradient

        #store the new w with the corresponding loss
        ws.append(next_w)
        losses.append(compute_loss_mse(y, tx, next_w))
        
        current_w = next_w
        
    #get index of w with the minimum loss
    best = np.argmin(losses)

    return ws[best], losses[best]

##################################################################################
################################  Least Square SGD ###############################
##################################################################################
def least_squares_SGD(y, tx, initial_w, max_iters = 500, gamma=1e-2, batch_size = 200):
    """
    Linear regression using stochastic gradient descent
    Args:
        y:  Target values 
        tx: Training data 
        initial_w:  initial w for the stochastic gradient descent
        max_iters:  maximum number of iterations, default_value=1000
        gamma:  step size
        batch_size: batch size for stochastic descent
    Returns:
        ws[best]:  Optimal weight vector
        losses[best]: MSE error of ws[best]
    """
    current_w = initial_w
    loss = compute_loss_mse(y, tx, current_w)

    #Store the initial_w with the corresponding loss
    ws = [current_w]
    losses = [loss]

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):

            stochastic_gradient = compute_gradient_mse(minibatch_y, minibatch_tx, current_w)
            
            #Update the w
            next_w = current_w - gamma*stochastic_gradient
            
            #Store current_w and loss
            ws.append(next_w)
            losses.append(compute_loss_mse(y, tx, next_w))

            current_w = next_w
            
    #get index of w with the minimum loss
    best = np.argmin(losses)
    
    return ws[best], losses[best]


##################################################################################
################################ Least Squares ###################################
##################################################################################
def least_squares(y, tx):
    """
    Least squares regression using normal equations
    Args:
        y:  Target values 
        tx: Training data
    Returns:
        w:  Optimal weight vector
        loss: MSE error of w   
    """
    AtA = np.dot(tx.T, tx)
    Aty = np.dot(tx.T, y)

    w = np.linalg.solve(AtA, Aty)
    # problem of singular matrix
    #w = np.linalg.lstsq(AtA, Aty)[0]
    loss = compute_loss_mse(y,tx,w)

    return w, loss



##################################################################################
################################  Ridge Regression ###############################
##################################################################################
def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    Args:
        y:  Target values 
        tx: Training data
        lambda_:  Penalizer
    Returns:
        w:  Optimal weight vector
        loss: MSE error of w
    """
    a = np.dot(tx.T,tx) + 2 * np.shape(tx)[0] * lambda_ * np.eye(np.shape(tx)[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_mse(y, tx, w)

    return w, loss


##################################################################################
################################  Logistic regression ############################
##################################################################################
def logistic_regression(y, tx, initial_w=None , max_iters=500, gamma=1e-3):
    """
    Logistic Regression using gradient descent
    Args:
        y:  Target values 
        tx: Training data
        initial_w:  initial w for the stochastic gradient descent
        max_iters:  maximum number of iterations, default_value=1000
        gamma:  step size
    Returns:
        ws[best]:  Optimal weights vector.
        losses[best]: Log-likelihood error at ws [best]
    """
    initial_w = np.zeros(tx.shape[1])
    current_w = initial_w
    loss = compute_loss_llh(y, tx, current_w)
    
    #Store the initial_w with the corresponding loss
    ws = [current_w]
    losses = [loss]
    
    for n_iter in range(max_iters): 
        gradient = compute_log_reg_gradient(y, tx, current_w)
        
        #Update w
        next_w = current_w - gamma*gradient
        
        #Store the new w with the corresponding loss
        ws.append(next_w)
        losses.append(compute_loss_llh(y, tx, next_w))

        current_w = next_w
        
    #get index of w with the minimum loss
    best = np.argmin(losses)

    return ws[best], losses[best]


##################################################################################
#####################  Regularized Logistic regression ###########################
##################################################################################
def reg_logistic_regression(y, tx, lambda_, initial_w=None, max_iters = 500, gamma = 1e-6):
    """
    Example function with PEP 484 type annotations.
    Args:
        y:  Target values 
        tx: Training data
        initial_w:  initial w for the stochastic gradient descent
        lambda_: penalty scalar        
        max_iters:  maximum number of iterations, default_=value=1000
        gamma:  step size
    Returns:
        ws[best]:  Optimal weights vector
        losses[best]: Log-likelihood error at ws [best]
    """
    initial_w = np.zeros(tx.shape[1])
    current_w = initial_w
    loss = compute_loss_llh(y, tx, current_w)

    #Store the initial_w with the corresponding loss
    ws = [current_w]
    losses = [loss]
    
    for n_iter in range(max_iters): 
        gradient = compute_log_reg_gradient(y, tx, current_w) + lambda_*current_w

        #Update w
        next_w = current_w - gamma*gradient

        #store the new w with the corresponding loss
        ws.append(next_w)
        losses.append(compute_loss_llh(y, tx, next_w))

        current_w = next_w
        
    #get index of w with the minimum loss
    best = np.argmin(losses)

    return ws[best], losses[best]

############################################################
######## Logistic Regression Helper Functions ##############
############################################################

def compute_log_reg_gradient(y, tx, w):
    '''Calculate the gradient of the log-likelihood function at w'''
    sig = compute_sigmoid(np.dot(tx, w))
    return np.dot(tx.T, sig-y)


def compute_sigmoid(z):
    '''Calculate the sigmoid value of z'''
    # Reformulate sigmoid function to prevent exponential overflow using adapt_sig  
    return 1 / (1 + np.exp(- adapt_sig(z)))


def adapt_sig(z):
    '''Reduce very large values to avoid the exponential overflow'''
    adapted_z = np.copy(z)
    adapted_z[z > 50] = 50
    adapted_z[z < -50] = -50
    return adapted_z


############################################################
################ Compute MSE loss ##########################
############################################################
def compute_loss_mse(y, tx, w):
    """
    Calculate the mse error.
    Args:
        y:  Target values 
        tx: Training data
        w:  Weights vector
    Returns:
        The mse error of w
    """
    e = y - tx.dot(w)
    return e.dot(e) / (2 * len(e))

############################################################
################ Compute Log-Likelihood loss ###############
############################################################
def compute_loss_llh(y, tx, w):
    """
    Compute the cost by negative log likelihood. 
    
    Args:
        y:  Target values 
        tx: Training data
        w:  Weights vector
    Returns:
        loss: The log-likelihood error of w
    """
    loss = np.sum( np.log(1+np.exp(adapt_sig(np.dot(tx,w)))) ) - np.dot(y.T, np.dot(tx,w))    
    return loss

############################################################
################ Compute the gradient of the MSE function###
############################################################
def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient
    Args:
        y:  Target values
        tx: Training data
        w:  Weights vector
    Returns:
        gradient: gradient of mean square error function at w
    """
    error = y - np.dot(tx, w)
    N = len(y)
    gradient = (-1/N)*np.dot(tx.T,error)
    return gradient


############################################################
################ Batch_iter ################################
############################################################
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

############################################################
######################### Build Polynomial #################
############################################################
def build_poly(tX, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    Returns the matrix formed by applying the polynomial basis to the input data
    """
    """polynomial basis functions for input data tx"""
    # returns the matrix formed by applying the polynomial basis to the input data
    poly = np.zeros((tX.shape[0],1+(tX.shape[1]-1)*degree))
    poly[:,0] = np.ones((tX.shape[0],))
    for deg in np.arange(1,degree+1):
        poly[:,1+(deg-1)*(tX.shape[1]-1):1+deg*(tX.shape[1]-1)] = tX[:,1:tX.shape[1]]**deg
    
    return poly


############################################################
######################### Compute Accuracy #################
############################################################
def compute_accuracy(y, tx, w):
    y_pred = predict_labels(w, tx)
    return np.mean(np.abs(y_pred - y)/2)