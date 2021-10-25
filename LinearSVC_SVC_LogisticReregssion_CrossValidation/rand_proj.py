import numpy as np

# Generate normal distribution matrix and get X1
def rand_proj (X, d):
    G = np.random.normal(loc=0.0, scale=1.0,size = (64,d))
    X1 = np.dot(X, G)
    return X1
