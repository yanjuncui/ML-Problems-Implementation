import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def split_results(X, i, split, inequal):
    pred = np.ones(len(X))
    if inequal == 'less':
        pred = np.where(X[:,i] <= split, 1, -1)
    else:
        pred = np.where(X[:,i] > split, 1, -1)
    return pred

def information_gain(X, y, w):
  pos = 0
  neg = 0
  for i in range(len(y)):
    if y[i] == 1:
      pos += w[i]
    else:
      neg += w[i]
  pos = pos/np.sum(w)
  neg = neg/np.sum(w)
  # print('---',pos, neg)
  entropy = -np.log2(pos)*pos - np.log2(neg)*neg
  cond_entro = np.zeros(len(X[0]))
  stumps = np.zeros(len(X[0]))
  error = np.zeros(len(X[0]))
  for i in range(0, len(X[0])):
    feature = X[:,i]
    uni = np.unique(feature)
    num_uni = np.zeros(len(uni))
    pos_uni = np.zeros(len(uni))

    for j in range(len(y)):
      index = np.argwhere(uni == feature[j])
      num_uni[index] += w[j]
      if y[j] == 1:
        pos_uni[index] += w[j]
    for j in range(len(uni)):
      if pos_uni[j] == 0:
        temp_entro = - (1-pos_uni[j]/num_uni[j])*np.log2(1-pos_uni[j]/num_uni[j])
      elif pos_uni[j] == num_uni[j]:
        temp_entro = -pos_uni[j]/num_uni[j]*np.log2(pos_uni[j]/num_uni[j])
      else:
        temp_entro = -pos_uni[j]/num_uni[j]*np.log2(pos_uni[j]/num_uni[j]) - (1-pos_uni[j]/num_uni[j])*np.log2(1-pos_uni[j]/num_uni[j])
      cond_entro[i] += temp_entro * (num_uni[j]/np.sum(w))
  max_IG = 1- np.min(cond_entro)
  best_col = np.argmin(cond_entro)
  return best_col

def decision_tree(X, y, w):
    steps = 10
    pred = np.zeros((1, len(y)))
    # stump, col_num, inequal, bestpred = 0,0,'less',np.zeros(len(y))
    min_error = 1
    # for i in range(len(X[0])):
    col_num = information_gain(X, y, w)

    min = np.min(X[:,col_num])
    max = np.max(X[:,col_num])
    step_size = (max-min)/steps
    for j in range(-1, int(steps+1)):
        for sign in ['less', 'greater']:
            split = min + j*step_size
            pred = split_results(X, col_num, split, sign)
            # print(split, pred)
            error = 0
            for m in range(len(X)):
                if pred[m] != y[m]:
                    error += w[m]
            if error < min_error:
                # print(error)
                min_error = error
                stump = split
                inequal = sign
                bestpred = pred
    # print(stump, col_num, inequal, min_error)
    return stump, col_num, inequal, min_error, bestpred

def fit(X, y, x_test, y_test):
    errorList = []
    testErrorList = []
    alpha = []
    w = np.ones(len(y))/len(y)
    gx = np.zeros((1, len(y)))
    stumps = []
    cols = []
    inequals = []
    for i in range(100):
        stump, col_num, inequal, error, pred = decision_tree(X, y, w)
        # print('-----', stump, col_num, inequal, error)
        stumps.append(stump)
        cols.append(col_num)
        inequals.append(inequal)
        cur_alpha = 1/2*np.log((1-error)/error)
        alpha.append(cur_alpha)
        w = w*np.exp(-cur_alpha*y*pred)
        z = np.sum(w)
        w = w/z
        gx += cur_alpha * pred
        cur_pred = np.where(gx > 0, 1, -1)
        cur_error = np.sum(cur_pred != y)/len(y)
        print('Weak learners number:', i, '. Training error is', cur_error)
        errorList.append(cur_error)

        test_res = predict(x_test, alpha, cols, stumps, inequals)
        cur_test_error = np.sum(test_res != y_test)/len(test_res)
        print('Weak learners number:', i, '. Testing error is', cur_test_error)
        testErrorList.append(cur_test_error)

    plt.title('Iteration times vs Training/Testing Error Rate')
    plt.plot(errorList,color = 'green', label = 'Training error')
    plt.plot(testErrorList, color = 'red', label = 'Test error')
    plt.legend()
    plt.show()

    return alpha, cols, stumps, inequals

def predict(x_test, alpha, cols, stumps, inequals):
    pred = np.zeros(len(x_test))
    for i in range(len(alpha)):
        cur_pred = split_results(x_test, cols[i], stumps[i], inequals[i])
        pred += alpha[i] * cur_pred
    pred = np.where(pred > 0, 1, -1)
    return pred


def adaboost(dataset):
    df = pd.read_csv(dataset, header = None)
    X = df.iloc[:, :10].values
    y = df.iloc[:, 10].values
    # X = np.delete(X, 6, axis = 1) # remove the features with missing data
    ques_index = np.where(X[:, 6] == '?')
    X = np.delete(X, ques_index, axis=0)
    X = X.astype(int)
    X = np.delete(X, 0, axis = 1)
    y = np.delete(y, ques_index, axis=0)
    y = np.where(y < 3, 1, -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    alpha, cols, stumps, inequals = fit(X_train, y_train, X_test, y_test)
    pred = predict(X_test, alpha, cols, stumps, inequals)
    error = np.sum(pred != y_test)/len(pred)
    print('Final predicting error is', error)


if __name__ == "__main__":
    adaboost('breast-cancer-wisconsin.data')
