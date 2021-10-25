from sklearn.linear_model import LogisticRegression
import numpy as np
import dataset
import my_cross_val
import warnings
from MySVM2 import MySVM2
warnings.filterwarnings("ignore")

data = [
[dataset.boston_x, dataset.boston50_y],
[dataset.boston_x, dataset.boston75_y],
]

method = [
MySVM2(d = np.shape(dataset.boston_x)[1], m = 40),
MySVM2(d = np.shape(dataset.boston_x)[1], m = 200),
MySVM2(d = np.shape(dataset.boston_x)[1], m = np.shape(dataset.boston_x)[0]),
LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)
]

model_name = ["MySVM2 with m = 40", "MySVM2 with m = 200", "MySVM2 with m = n", "LogisticRegression"]

data_name = ["Boston50", "Boston75"]

def q3():
    # my_cross_val.my_cross_val(method[0], data[1][0], data[1][1], 5)
    for i in range(0, 2):
        for j in range(0, 4):
            print("Error rates for "+model_name[j]+" for "+data_name[i])
            my_cross_val.my_cross_val(method[j], data[i][0], data[i][1], 5)


if __name__ == "__main__":
    q3()
