import pandas as pd
import io
import numpy as np
from cvxopt import solvers
from cvxopt import matrix
from sklearn.model_selection import train_test_split



C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]
sigma = [0.1, 1, 10, 100]

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
        index  = 0
        for j in range(num):
            for i in range(num):
                temp += lam[i,0]*y_train[i]*np.inner(x_train[i], x_train[j])
            # print(temp)
        b = (np.sum(y_train) - temp)/num

        for i in range(len(x_test)):
            pred = 0
            for j in range(len(x_train)):
                pred += lam[j,0]*y_train[j]*np.dot(x_test[i], x_train[j])
            if pred > 0:
                # print(pred)
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        # print("---", y_pred)
        return y_pred


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
        # for i in range(num):
        #     temp += lam[i,0]*y_train[i]*RBF_kernel.RBF(x_train[i], x_train[index], sigma)
        #     # print(temp)
        # b = y_train[index] - temp
        # print(temp, b)
        for i in range(num):
            for j in range(num):
                temp += lam[i,0]*y_train[i]*RBF_kernel.RBF(x_train[i], x_train[j], sigma)
            # print(temp)
        b = (np.sum(y_train) - temp)/num

        y_pred = np.zeros(len(x_test))
        for i in range(len(x_test)):
            pred = 0
            for j in range(len(x_train)):
                pred += lam[j,0]*y_train[j]*RBF_kernel.RBF(x_train[j], x_test[i], sigma)
            pred = pred + b
            if pred > 0:
                # print(pred)
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        # print("---", y_pred)
        return y_pred


def kernel_SVM(dataset:str):
    df = pd.read_csv(dataset)
    dataset = df.values
    labels = dataset[:1000, -1]
    for i in range(1000):
        if labels[i] == 0:
            labels[i] = -1
    features = dataset[:1000, :-1]
    best_c, best_error = 0, 1

    print("Begin of linear kernel: ")
    for j in range(len(C)):
        scores1 = np.zeros(10)
        scores2 = np.zeros(10)
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
            lam = linear_kernel.fit(C[j], X_train, y_train)
            y_pred1 = linear_kernel.predict(lam, X_test, X_train, y_train)
            error1 = 1- np.sum(y_pred1==y_test)/len(y_test)
            scores1[i] = error1
            # print("While C = ",C[j], "error for fold", i,"is", error)

            y_pred2 = linear_kernel.predict(lam, X_train, X_train, y_train)
            error2  = 1- np.sum(y_pred2==y_train)/len(y_train)
            scores2[i] = error2
        print("Validation: While C = ", C[j], "mean of error is", scores2.mean(), "and std of error is ", scores2.std())
        print("Train: While C = ", C[j], "mean of error is", scores1.mean(), "and std of error is ", scores1.std())
        print("-----")
        if best_error > scores1.mean():
            best_error = scores1.mean()
            best_c = C[j]
    print("Conclusion: For linear kernel, Best C and Lowest error rate is: ", best_c, ", ", best_error)

    best_c, best_error, best_sigma = 0, 1, 0
    print("=========================")
    print("Begin of RBF kernel: ")
    for s in range(len(sigma)):
        for j in range(len(C)):
            scores1 = np.zeros(10)
            scores2 = np.zeros(10)
            for i in range(10):
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
                lam = RBF_kernel.fit(C[j], sigma[s], X_train, y_train)
                y_pred1 = RBF_kernel.predict(lam, X_test, sigma[s], X_train, y_train)
                error1 = 1- np.sum(y_pred1==y_test)/len(y_test)
                scores1[i] = error1
                # print("Train: While C = ",C[j], "error for fold", i,"is", error1)

                y_pred2 = RBF_kernel.predict(lam, X_train, sigma[s], X_train, y_train)
                error2  = 1- np.sum(y_pred2==y_train)/len(y_train)
                scores2[i] = error2
                # print("Validation: While C = ",C[j], "error for fold", i,"is", error2)

            print("Validation: While sigma =", sigma[s], "C = ", C[j], "mean of error is", scores2.mean(), "and std of error is ", scores2.std())
            print("Train: While sigma =", sigma[s], "C = ", C[j], "mean of error is", scores1.mean(), "and std of error is ", scores1.std())
            print("-----")
            if best_error > scores1.mean():
                best_error = scores1.mean()
                best_c = C[j]
                best_sigma = sigma[s]
    print("Conclusion: For RBF kernel, Best C ,sigma, and Lowest error rate is: ", best_c, ", ", best_sigma, ", ", best_error)

if __name__ == "__main__":
    kernel_SVM('hw2_data_2020.csv')
