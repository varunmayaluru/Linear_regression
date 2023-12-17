# import required libraries
import numpy as np
import pandas as pd

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

# single prediction element by element
def single_prediction(X, w, b):
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p 
    """
    n = X.shape[0]
    print('n: ', n)
    p = 0
    for i in range(n):
        p += X[i] * w[i]
    p += b
    return p

# if X_train and w_init are different shape, then we need to transpose X_train
def multiple_prediction(X, w, b):
    """
    multiple predict using linear regression
    
    Args:
      x (ndarray): Shape (n, m) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p 
    """
    n = X.shape[0]
    m = X.shape[1]
    print('n: ', n)
    print('m: ', m)
    p = np.zeros((n, 1))
    for i in range(n):
        for j in range(m):
            p[i] += X[i][j] * w[j]
        p[i] += b
    return p

# vectorized implementation
def multiple_prediction_vectorized(X, w, b):
    """
    multiple predict using linear regression
    
    Args:
      x (ndarray): Shape (n, m) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p 
    """
    n = X.shape[0]
    m = X.shape[1]
    print('n: ', n)
    print('m: ', m)
    p = np.zeros((n, 1))
    for i in range(n):
        p[i] = np.dot(X[i], w) + b
    return p

# create a function to compute the cost
def compute_cost(X, y, w, b):
    """
    compute cost for linear regression
    
    Args:
      x (ndarray): Shape (n, m) example with multiple features
      y (ndarray): Shape (n,) example labels    
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      cost (scalar):  model cost     
    """
    n = X.shape[0]
    m = X.shape[1]
    cost = 0
    for i in range(n):
        cost += (y[i] - (np.dot(X[i], w) + b)) ** 2
    cost /= n
    return cost

# create a function to compute gradient

def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

# create a function for gradient descent

def gradient_descent(X, y, w, b, alpha, num_iterations):
    """
    gradient descent for linear regression

    Args:
      x (ndarray): Shape (n, m) example with multiple features
      y (ndarray): Shape (n,) example labels    
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter
      alpha (scalar): learning rate
      num_iterations (int): number of iterations

    Returns:
        w (ndarray): Shape (n,) model parameters    
        b (scalar):  model parameter
        cost_history (list): cost history
    """
    cost_history = []
    for i in range(num_iterations):
        dw, db = compute_gradient(X, y, w, b)
        w += alpha * dw
        b += alpha * db
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
    return w, b, cost_history


if __name__ == '__main__':
    # single prediction
    X = np.array([2104, 5, 1, 45])
    p = single_prediction(X, w_init, b_init)
    print('Single prediction: ', p)
    # multiple prediction
    X = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    p = multiple_prediction(X, w_init, b_init)
    print('Multiple prediction: ', p)
    # compute cost
    cost = compute_cost(X_train, y_train, w_init, b_init)
    print('Cost: ', cost)
    # compute gradient
    db, dw = compute_gradient(X_train, y_train, w_init, b_init)
    print('dw: ', dw)
    print('db: ', db)

