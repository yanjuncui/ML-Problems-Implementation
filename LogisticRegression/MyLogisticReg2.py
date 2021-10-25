import numpy as np

class MyLogisticReg2:
    def __init__(self, d):
        self.d = d
        self.step_size = 0.01 # step size
        self.iter = 500 #iteration time
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

    def predict(self, X):
        ypred = []
        for i in range(0, len(X)):
            sig_y = 1/(1+np.exp(-(np.dot(self.w[1:14], X[i])+self.w[0])))
            if sig_y > 0.5:
                ypred.append(1)
            else:
                ypred.append(0)
        return ypred
