import dataset
import my_cross_val
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = [
[dataset.boston_x, dataset.boston50_y],
[dataset.boston_x, dataset.boston75_y],
[dataset.digit_x, dataset.digit_y]
]
class LogReg1d:
    def __init__(self, d):
        self.d = d
        self.step_size = 0.001 # step size
        self.iter = 300 #iteration time
        self.w = np.random.uniform(-0.01,0.01,14) # self.w[0] saves w0

    def fit(self, X, y):
        for repeat in range(0, self.iter):
            delta_w = np.zeros(14) # delta_w[0] saves delta_w[0] for changing w0

            for i in range(0, len(X)):
                o = self.w[0]
                o = o + np.dot(self.w[1:14].T, X[i])
                # for j in range(1, 14):
                #     o = o + (self.w[j]) * X[i][j-1]
                sig_y = 1/(1+np.exp(-o))

                delta_w[0] = delta_w[0] +(y[i]-sig_y) * 1
                for j in range(1, 14):
                    delta_w[j] = delta_w[j] + (y[i]-sig_y) * X[i][j-1]

            for j in range(0, 14):
                self.w[j] = self.w[j]+ self.step_size * delta_w[j]
        # print(self.w)

    def predict(self, X):
        ypred = []
        for i in range(0, len(X)):
            sig_y = 1/(1+np.exp(-(np.dot(self.w[1:14], X[i])+self.w[0])))
            if sig_y > 0.5:
                ypred.append(1)
            else:
                ypred.append(0)
        return ypred

class LogReg2d:
    def __init__(self, k ,d):
        self.d = d
        self.k = 10
        self.step_size = 0.0001 # step size
        self.iter = 40 #iteration time
        self.w = np.random.uniform(-0.01,0.01,d*k) # self.w[:,0] saves w0
        self.w = self.w.reshape((k,d))

    def fit(self, X, y):
        for repeat in range(0, self.iter):
            # print(self.k, self.d)
            delta_w = np.zeros((self.k, self.d))
            r = np.zeros((len(X), self.k))
            for t in range(len(X)):
                r[t, int(y[t])] = 1

            for t in range(len(X)):
                o = np.zeros(self.k)
                for i in range(self.k):
                    # o[i] = self.w[i][0]
                    for j in range(0, self.d):
                        o[i,] += self.w[i][j]*X[t][j]
                sum = 0
                for i in range(self.k):
                    sum += np.exp(o[i])
                y_pred = np.exp(o)/sum
                for i in range(self.k):

                    for j in range(self.d):
                        delta_w[i,j] = delta_w[i,j] + (r[t, i]-y_pred[i])*X[t][j]
                # print(delta_w)
            self.w = self.w + self.step_size*delta_w
            #print(self.w)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        o = np.dot(X, self.w.T)
        for i in range(len(X)):
            y_pred[i] = np.argmax(o[i])
        return y_pred


def logisticRegression(num_splits, train_percent):
    for i in range(0, 2):
        # print("-------")
        if i==0:
            print("Boston50 dataset: ")
        else:
            print("==========")
            print("boston75 dataset: ")
        method1 = LogReg1d(d = np.shape(data[i][0])[1])
        my_cross_val.my_cross_val_q4(method1, data[i][0],data[i][1], num_splits, train_percent)
    print("==========")
    print("Digit dataset:")
    method2 = LogReg2d(k = 10, d = np.shape(data[2][0])[1])
    my_cross_val.my_cross_val_q4(method2, data[2][0],data[2][1], num_splits, train_percent)

if __name__ == "__main__":
    logisticRegression(10, [10, 25, 50, 75, 100])
