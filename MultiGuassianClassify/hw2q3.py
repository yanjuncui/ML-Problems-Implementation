from sklearn.linear_model import LogisticRegression
import numpy as np
import dataset
import my_cross_val
import warnings
from MultiGaussClassify import MultiGaussClassify
warnings.filterwarnings("ignore")

data = [
[dataset.boston_x, dataset.boston50_y],
[dataset.boston_x, dataset.boston75_y],
[dataset.digit_x, dataset.digit_y]
]

# K is the number of class in dataset, D is dimensionality of features

method = [
MultiGaussClassify(k = len(np.unique(dataset.boston50_y)), d = np.shape(dataset.boston_x)[1]),
MultiGaussClassify(k = len(np.unique(dataset.boston75_y)), d = np.shape(dataset.boston_x)[1]),
MultiGaussClassify(k = len(np.unique(dataset.digit_y)), d = np.shape(dataset.digit_x)[1]),
LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)
]

model_name = ["MultiGaussClassify with full covarience matrix",
              "MultiGaussClassify with diagonal covarience matrix",
              "LogisticRegression"]

data_name = ["Boston50", "Boston75", "Digits"]



def hw2q3():
    for i in range(0, 3):
        print("Error rates for "+model_name[0]+" on "+data_name[i])
        my_cross_val.my_cross_val2(method[i], data[i][0], data[i][1], 5, False)
    for i in range(0, 3):
        print("Error rates for "+model_name[1]+" on "+data_name[i])
        my_cross_val.my_cross_val2(method[i], data[i][0], data[i][1], 5, True)
    for i in range(0, 3):
        print("Error rates for "+model_name[2]+" on "+data_name[i])
        my_cross_val.my_cross_val(method[3], data[i][0], data[i][1], 5)

if __name__ == "__main__":
    hw2q3()
