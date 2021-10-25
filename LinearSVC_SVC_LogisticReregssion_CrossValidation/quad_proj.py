import numpy as np

# get dataset for X2
# X2 contains three parts, its own (X), the the square of every elements (x_part2)
# and the combination with other features(x_part3)
def quad_proj(X):
    x_part2 = np.multiply(X, X)
    x_part3 = []
    for i in range(0, len(X)):
        helper = []
        for j in range(0, len(X[0])):
            for m in range(j+1, len(X[0])):
                helper.append(X[i][j]*X[i][m])
        helper = np.array(helper)
        x_part3.append(helper)
    x_part3 = np.array(x_part3)
    X2 = np.hstack((X, x_part2, x_part3))
    X2 = np.array(X2)
    # print(X2)
    return X2
