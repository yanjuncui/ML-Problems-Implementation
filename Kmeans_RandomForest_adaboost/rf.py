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

def get_subsample(X, y):
  state = np.random.get_state()
  np.random.shuffle(X)
  np.random.set_state(state)
  np.random.shuffle(y)
  sub_X = X[:50, ]
  sub_y = y[:50]
  return sub_X, sub_y

def classification(X, y, stumps, cols, inequals):
  total_pred = np.zeros((len(stumps), len(y)))
  for i in range(len(stumps)):
    total_pred[i] = split_results(X, cols[i], stumps[i], inequals[i])
  res = np.ones(len(y))
  for i in range(len(y)):
    pos = np.sum(total_pred[:,i] == 1)
    if pos < 0.5 *len(stumps):
      res[i] = -1
  return res


def rf(dataset):
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
    w = np.ones(len(y))/len(y)
    stumps = []
    cols = []
    inequals = []
    test_feature_error = []
    train_feature_error = []
    for m in range(2, len(X[0])+1):
        trainErrList = []
        testErrorList = []
        for i in range(100):
            col = np.arange(len(X_train[0]))
            sub_col = np.random.choice(col, m)
            sub_X, sub_y = get_subsample(X_train, y_train)
            stump, col_num, inequal, min_error, bestpred = decision_tree(sub_X[:, sub_col], sub_y, w)
            # print(stump, col_num, inequal, min_error)
            stumps.append(stump)
            cols.append(col_num)
            inequals.append(inequal)
            train_pred = classification(X_train, y_train, stumps, cols, inequals)
            test_pred = classification(X_test, y_test, stumps, cols, inequals)
            # print(pred)
            train_error = np.sum(y_train!=train_pred)/len(y_train)
            test_error = np.sum(y_test!=test_pred)/len(y_test)
            trainErrList.append(train_error)
            testErrorList.append(test_error)
            if i == 99:
                test_feature_error.append(test_error)
                train_feature_error.append(train_error)

        if m == 3:
            print('Training error rates with m = 3 is', trainErrList)
            print('Testing error rates with m = 3 is', testErrorList)

            plt.title('Iteration times vs Training/Testing Error Rate')
            plt.plot(trainErrList,color = 'green', label = 'Training error')
            plt.plot(testErrorList, color = 'red', label = 'Test error')
            plt.legend()
            plt.show()

    print('Training error rates with m = 2 to m = 9 is', train_feature_error)
    print('Testing error rates with m = 2 to m = 9 is', test_feature_error)

    plt.title('Number of features vs Training/Testing Error Rate')
    x = [2, 3, 4, 5, 6, 7, 8, 9]
    plt.plot(x, train_feature_error,color = 'blue', label = 'Training error')
    plt.plot(x, test_feature_error, color = 'yellow', label = 'Test error')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    rf('breast-cancer-wisconsin.data')
