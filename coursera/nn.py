import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import math

def softmax(Z):
    Z_minus = Z - np.max(Z, axis=0) + 0.001
    cache = Z

    return np.exp(Z_minus) / np.sum(np.exp(Z_minus), axis=0, keepdims=True), cache

def softmax_backward(dA, cache):
    Z = cache

    Z_minus = Z - np.max(Z, axis=0) + 0.001
    s = np.exp(Z_minus) / np.sum(np.exp(Z_minus), axis=0, keepdims=True)

    return dA * s * (1. - s)

def relu(Z):
    cache = Z
    return np.maximum(0., Z), cache

def relu_backward(dA, cache):
    Z = cache

    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):    
    if activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "softmax")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = -1 / m * np.sum(Y * np.log(AL + 0.001))
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(cache[1].T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    (linear_cache, activation_cache) = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # to softmax
    dAL = -np.divide(Y, AL + 0.0001)

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "softmax")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - grads["dW" + str(l + 1)] * learning_rate
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - grads["db" + str(l + 1)] * learning_rate

    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = []                         # keep track of cost
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 0
        else:
            p[0,i] = 1
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((probas.argmax(0) == y.argmax(0))/m)))
        
    return p

if __name__ == '__main__':
    df = pd.read_csv('data.csv')

    attirbutes = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
                'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    columns = ['ID', 'Diagnosis']
    for s in attirbutes:
        columns.append(f'Mean {s}')
        columns.append(f'{s} SE')
        columns.append(f'Worst {s}')
    df.columns = columns

    # features = ['Worst Area', 'Worst Smoothness', 'Mean Texture']
    # X = df[features].values
    train_x = df.drop(columns=['ID', 'Diagnosis']).values
    train_x = (train_x - train_x.mean()) / train_x.std()

    oheDiagnosis = OneHotEncoder()
    train_y = oheDiagnosis.fit_transform(df[['Diagnosis']]).toarray()

    n_layers = train_x.shape[1]

    layers_dims = [n_layers, 30, 30, 2] #  4-layer model

    parameters, costs = L_layer_model(train_x.T, train_y.T, layers_dims, num_iterations = 8000, print_cost = True)
    predict(train_x.T, train_y.T, parameters)
