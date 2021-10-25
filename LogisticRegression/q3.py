from sklearn.linear_model import LogisticRegression
import numpy as np
import dataset
import my_cross_val
import warnings
from MyLogisticReg2 import MyLogisticReg2
warnings.filterwarnings("ignore")

data = [
[dataset.boston_x, dataset.boston50_y],
[dataset.boston_x, dataset.boston75_y],
]

method = [
MyLogisticReg2(d = np.shape(dataset.boston_x)[1]),
LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)
]

model_name = ["MyLogisticReg2", "LogisticRegression"]

data_name = ["Boston50", "Boston75"]

def q3():
    for i in range(0, 2):
        for j in range(0, 2):
            print("Error rates for "+model_name[i]+" on "+data_name[j])
            my_cross_val.my_cross_val(method[i], data[j][0], data[j][1], 5)


if __name__ == "__main__":
    q3()
