import pandas as pd
import io
import numpy as np
from cvxopt import solvers
from cvxopt import matrix
from sklearn.model_selection import train_test_split

C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]
sigma = [10, 100, 500, 1000]


class linear_kernel:
    def fit(c, X, y):
        num = len(y)
        M = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                M[i,j] = y[i]*y[j]*np.inner(X[i],X[j])
                # print(H[i,j])
        P = matrix(M)
        q = matrix(-np.ones((num, 1)))
        G = matrix(np.append(np.eye(num)*-1, np.eye(num)).reshape(2*num, num))
        h = matrix(np.append(np.zeros((num, 1)), np.ones((num, 1))*c))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        lam = np.array(sol['x'])
        return lam

    def predict(lam, x_test, x_train, y_train):
        y_pred = np.zeros(len(x_test))
        num = len(x_train)
        temp = 0
        for j in range(num):
            for i in range(num):
                temp += lam[i,0]*y_train[i]*np.inner(x_train[i], x_train[j])
            # print(temp)
        b = (np.sum(y_train) - temp)/num
        # index  = 0
        # for i in range(num):
        #     temp += lam[i,0]*y_train[i]*np.inner(x_train[i], x_train[index])
        #     # print(temp)
        # b = y_train[index] - temp

        for i in range(len(x_test)):
            pred = 0
            for j in range(len(x_train)):
                pred += lam[j,0]*y_train[j]*np.dot(x_test[i], x_train[j])
            y_pred[i] = pred + b
        # print("---", y_pred)
        return y_pred

def multi_linear_predict(c, x_train, y_train, x_test, y_test):
    num = len(y_train)
    k =  len(np.unique(y_train))
    # print(k)
    y_pred1 = np.zeros((len(x_test), k))
    y_pred2 = np.zeros((len(x_train), k))
    new_y_train = np.zeros(num)
    for i in range(k):
        for j in range(num):
            if y_train[j] == i:
                new_y_train[j] = 1
            else:
                new_y_train[j] = -1

        lam = linear_kernel.fit(c, x_train, new_y_train)
        y_pred_test = linear_kernel.predict(lam, x_test, x_train, new_y_train)
        y_pred_train = linear_kernel.predict(lam, x_train, x_train, new_y_train)
        for j in range(len(x_test)):
            if y_pred_test[j] > 0:
                y_pred1[j,i] = y_pred_test[j]
        # print(y_pred1)
        for j in range(len(x_train)):
            if y_pred_train[j] > 0:
                y_pred2[j,i] = y_pred_train[j]
        # print(y_pred)
    res1 = np.argmax(y_pred1, axis = 1)
    res2 = np.argmax(y_pred2, axis = 1)
    # print(res1, res2)
    return res1, res2

class RBF_kernel:
    def RBF(x1, x2, sigma):
        return np.exp(-np.linalg.norm(x1-x2)**2/(2*(sigma**2)))

    def fit(c, s, X, y):
        num = len(y)
        M = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                 M[i,j] = y[i]*y[j]*RBF_kernel.RBF(X[i], X[j], s)
                 #print(M[i,j])
        P = matrix(M)
        q = matrix(-np.ones((num, 1)))
        G = matrix(np.append(np.eye(num)*-1, np.eye(num)).reshape(2*num, num))
        h = matrix(np.append(np.zeros((num, 1)), np.ones((num, 1))*c))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        lam = np.array(sol['x'])
        return lam

    def predict(lam, x_test, sigma, x_train, y_train):
        num = len(y_train)
        temp = 0
        index  = 0
        for i in range(num):
            for j in range(num):
                temp += lam[i,0]*y_train[i]*RBF_kernel.RBF(x_train[i], x_train[j], sigma)
            # print(temp)
        b = (np.sum(y_train) - temp)/num
        # for i in range(num):
        #     temp += lam[i,0]*y_train[i]*RBF_kernel.RBF(x_train[i], x_train[index], sigma)
        #     # print(temp)
        # b = y_train[index] - temp
        # print("b", b)
        y_pred = np.zeros(len(x_test))
        for i in range(len(x_test)):
            pred = 0
            for j in range(num):
                pred += lam[j,0]*y_train[j]*RBF_kernel.RBF(x_train[j], x_test[i], sigma)
                # print("----",pred)
            y_pred[i] = pred + b
            # print("---", pred, y_pred[i])
        return y_pred

def multi_RBF_predict(c, s, x_train, y_train, x_test, y_test):
    num = len(y_train)
    k =  len(np.unique(y_train))
    # print(k)
    y_pred1 = np.zeros((len(x_test), k))
    y_pred2 = np.zeros((len(x_train), k))
    new_y_train = np.zeros(num)
    for i in range(k):
        for j in range(num):
            if y_train[j] == i:
                new_y_train[j] = 1
            else:
                new_y_train[j] = -1
        lam = RBF_kernel.fit(c, s, x_train, new_y_train)

        y_pred_test = RBF_kernel.predict(lam, x_test, s, x_train, new_y_train)
        y_pred_train = RBF_kernel.predict(lam, x_train, s, x_train, new_y_train)

        for j in range(len(x_test)):
            if y_pred_test[j] > 0:
                y_pred1[j,i] = y_pred_test[j]
        # print(y_pred1)
        for j in range(len(x_train)):
            if y_pred_train[j] > 0:
                y_pred2[j,i] = y_pred_train[j]

    res1 = np.argmax(y_pred1, axis = 1)
    res2 = np.argmax(y_pred2, axis = 1)
    # print(res1, res2)
    return res1, res2

def get_error(a, b):
    total = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            total += 1
    return 1 - total/len(a)

def multi_SVM(dataset:str):
    # read and dealing data into numpy
    df = pd.read_csv(dataset+"/mfeat-fac", header = None, delimiter = r"\s+")
    dataset1 = df.values
    df = pd.read_csv(dataset+"/mfeat-fou", header = None, delimiter = r"\s+")
    dataset2 = df.values
    df = pd.read_csv(dataset+"/mfeat-kar", header = None, delimiter = r"\s+")
    dataset3 = df.values
    df = pd.read_csv(dataset+"/mfeat-mor", header = None, delimiter = r"\s+")
    dataset4 = df.values
    df = pd.read_csv(dataset+"/mfeat-pix", header = None, delimiter = r"\s+")
    dataset5 = df.values
    df = pd.read_csv(dataset+"/mfeat-zer", header = None, delimiter = r"\s+")
    dataset6 = df.values
    dataset = np.concatenate([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6], axis = 1)
    # print(np.shape(dataset))

    label0 = np.zeros((50, 1))
    label1 = np.ones((50, 1))
    label2 = np.array([[2]*50]).reshape((50, 1))
    label3 = np.array([[3]*50]).reshape((50, 1))
    label4 = np.array([[4]*50]).reshape((50, 1))
    label5 = np.array([[5]*50]).reshape((50, 1))
    label6 = np.array([[6]*50]).reshape((50, 1))
    label7 = np.array([[7]*50]).reshape((50, 1))
    label8 = np.array([[8]*50]).reshape((50, 1))
    label9 = np.array([[9]*50]).reshape((50, 1))
    # print(label2)
    labels = np.concatenate([label0, label1, label2, label3, label4, label5, label6, label7, label8, label9], axis = 0)
    # print(np.shape(labels))
    features = np.array([])
    for i in range(10):
        features = np.append(features, dataset[i*200: i*200+50])
    features = features.reshape((500, len(dataset[0])))
    # print(np.shape(features))

    best_c, best_error = 0, 1
    print("Begin of multi linear kernel: ")
    for j in range(len(C)):
        scores1 = np.zeros(10)
        scores2 = np.zeros(10)
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
            y_pred1, y_pred2 = multi_linear_predict(C[j], X_train, y_train, X_test, y_test)
            error1 = get_error(y_pred1, y_test)
            scores1[i] = error1
            # print("While C = ",C[j], "error for fold", i,"is", error1)

            # y_pred2 =  multi_linear_predict(C[j], X_train, y_train, X_train, y_train)
            error2 = get_error(y_pred2, y_train)
            scores2[i] = error2
            # print("While C = ",C[j], "error for fold", i,"is", error2)

        print("Validation: While C = ", C[j], "mean of error is", scores2.mean(), "and std of error is ", scores2.std())
        print("Train: While C = ", C[j], "mean of error is", scores1.mean(), "and std of error is ", scores1.std())
        print("-----")
        if best_error > scores1.mean():
            best_error = scores1.mean()
            best_c = C[j]
    print("Conclusion: For linear kernel, Best C and Lowest error rate is: ", best_c, ", ", best_error)

    print("=========================")
    print("Begin of multi RBF kernel: ")
    for s in range(len(sigma)):
        for j in range(len(C)):
            scores1 = np.zeros(10)
            scores2 = np.zeros(10)
            for i in range(10):
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

                y_pred1, y_pred2 = multi_RBF_predict(C[j], sigma[s], X_train, y_train, X_test, y_test)
                error1 = get_error(y_pred1, y_test)
                scores1[i] = error1
                # print("Train: While C = ",C[j], "sigma = ", sigma[s], ", error for fold", i,"is", error1)

                # y_pred2 = multi_RBF_predict(C[j], sigma[s], X_train, y_train,  X_train, y_train)
                error2  = get_error(y_pred2, y_train)
                scores2[i] = error2
                # print("Validation: While C = ",C[j], "sigma = ", sigma[s], ", error for fold", i,"is", error2)

            print("Validation: While sigma =", sigma[s], "C = ", C[j], "mean of error is", scores2.mean(), "and std of error is ", scores2.std())
            print("Train: While sigma =", sigma[s], "C = ", C[j], "mean of error is", scores1.mean(), "and std of error is ", scores1.std())
            print("-----")
            if best_error > scores1.mean():
                best_error = scores1.mean()
                best_c = C[j]
                best_sigma = sigma[s]
    print("Conclusion: For RBF kernel, Best C ,sigma, and Lowest error rate is: ", best_c, ", ", best_sigma, ", ", best_error)

if __name__ == "__main__":
    multi_SVM('mfeat')
