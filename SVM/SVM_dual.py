import pandas as pd
import io
import numpy as np
from cvxopt import solvers
from cvxopt import matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]

class Dual_SVM:
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
        w = np.zeros(len(X[0]))
        for i in range(num):
          for j in range(len(X[0])):
            w[j] += y[i]*lam[i][0]*X[i][j]

        temp = 0
        index  = 0
        for i in range(num):
            temp += np.inner(w, X[i])
            # print(temp)
        b = (np.sum(y) - temp)/num
        # print(b)
        return w, b

    def predict(w, b, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            if np.dot(w,X[i])+b > 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        # print(y_pred)
        return y_pred


def SVM_dual(dataset:str):
    df = pd.read_csv(dataset)
    dataset = df.values
    labels = dataset[:1000, -1]
    for i in range(1000):
        if labels[i] == 0:
            labels[i] = -1
    features = dataset[:1000, :-1]
    best_c, best_error = 0, 1

    for j in range(len(C)):
        scores1 = np.zeros(10)
        scores2 = np.zeros(10)
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
            w, b = Dual_SVM.fit(C[j], X_train, y_train)
            y_pred1 = Dual_SVM.predict(w, b, X_test)
            # score = cross_val_score(Dual_SVM, features, labels, cv=10)
            error1 = 1- np.sum(y_pred1==y_test)/len(y_test)
            # print("While C = ",C[j], "error for fold", i,"is", error)
            scores1[i] = error1
            y_pred2 = Dual_SVM.predict(w, b, X_train)
            error2  = 1- np.sum(y_pred2==y_train)/len(y_train)
            scores2[i] = error2
        print("Validation: For C = ", C[j], "mean of error is", scores2.mean(), "and std of error is ", scores2.std())
        print("Train: For C = ", C[j], "mean of error is", scores1.mean(), "and std of error is ", scores1.std())
        print("-----")
        if best_error > scores1.mean():
            best_error = scores1.mean()
            best_c = C[j]
    print("Conclusion: Best C and Lowest error rate is: ", best_c, ", ", best_error)

if __name__ == "__main__":
    SVM_dual('hw2_data_2020.csv')
