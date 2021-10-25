import numpy as np

def get_fw(w, dataset):
    fw1 = 0
    fw2 = 0
    for i in range(1, 14):
        fw1 = fw1 + w[i]*w[i]
    for i in range(0, len(dataset)):
        m = 1 - (np.dot(w[1:14], dataset[i][0])+w[0])*dataset[i][1]
        if m > 0:
            fw2 = fw2 + m

    fw = fw2/len(dataset) + 5/2*fw1
    return fw

# randomly select m elements in n elements
def get_batch(x, y, m, n):
    rand_data = []

    for i in range(0, n):
        rand_data.append((x[i], y[i]))

    np.random.shuffle(rand_data)
    mini = rand_data[0:m]
    return rand_data, mini


class MySVM2:
    def __init__(self, d, m):
        self.d = d
        self.step_size = 0.1 # step size
        self.iter = 200 #iteration time
        # self.w = np.random.uniform(-0.01,0.01,14) # self.w[0] saves w0
        self.w = np.random.uniform(-0.1,0.1,13)
        self.b = 0
        #self.final_w = self.w
        self.final_fw = float("inf")
        self.m = m

    def fit(self, X, y):
        for repeat in range(0, self.iter):
            rand_data, sub_data = get_batch(X, y, self.m, len(y))
            # print(rand_data)
            delta_w = np.zeros(14) # delta_w[0] saves delta_w[0] for changing w0
            if self.m > len(y):
                self.m = len(y)

            for i in range(0, self.m):
                o = self.w[0]
                o = o + np.dot(self.w[1:14].T, sub_data[i][0])

                sig_y = 1/(1+np.exp(-o))

                delta_w[0] = delta_w[0] +(sub_data[i][1]-sig_y) * 1
                for j in range(1, 14):
                    delta_w[j] = delta_w[j] + (sub_data[i][1]-sig_y) * sub_data[i][0][j-1]
                for j in range(0, 14):
                    self.w[j] = self.w[j] + self.step_size * delta_w[j]

            fw = get_fw(self.w, sub_data)
            # print(fw)
            if self.final_fw > fw:
                self.final_fw = fw
                self.final_w = self.w

    def predict(self, X):
        ypred = []
        for i in range(0,len(X)):
            pred_y = np.dot(self.final_w[1:14].T, X[i]) + self.final_w[0]
            # pred_y = 1/(1+np.exp(-(np.dot(self.w[1:14], X[i])+self.w[0])))
            # print(pred_y)
            if pred_y > 0:
                ypred.append(1)
            else:
                ypred.append(-1)
        return ypred
