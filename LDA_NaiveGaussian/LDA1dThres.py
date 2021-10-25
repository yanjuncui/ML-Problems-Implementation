
import dataset
import my_cross_val
import numpy as np

class LDA1d:
    # def __init__(self):
        # self.w = 0

    def fit(X, y):
        x1 = []
        x2 = []
        for i in range(len(X)):
            if y[i] == 1:
                x1.append(X[i])
            else:
                x2.append(X[i])
        x1 = np.array(x1)
        x2 = np.array(x2)

        m1 = np.mean(x1, axis = 0)
        m2 = np.mean(x2, axis = 0)
        # print(m1,m2)
        sw = np.dot((x1-m1).T,(x1-m1))+np.dot((x2-m2).T, (x2-m2))

        w = np.dot(np.linalg.inv(sw), (m1 - m2).reshape((len(m1), 1)))

        c1 = np.dot((m1 - m2).reshape(1, (len(m1))), np.linalg.inv(sw))
        c2 = np.dot(c1, (m1 + m2).reshape((len(m1), 1)))
        w0 = 1/2 * c2
        # print(c)
        return w, w0

    def predict(X, w, w0):

        ypred = []
        for i in range(0, len(X)):
            y = np.dot(w.T, X[i]) - w0
            # print(y)
            if y >= 0:
                ypred.append(1)
            else:
                ypred.append(0)
        return ypred

def LDA1dThres(num_crossval):
    my_cross_val.my_cross_val_q31(LDA1d, dataset.boston_x, dataset.boston50_y, num_crossval)

if __name__ == "__main__":
    LDA1dThres(10)
