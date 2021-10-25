from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import dataset
import my_train_test
import warnings
warnings.filterwarnings("ignore")

method = [
LinearSVC(max_iter = 2000),
SVC(gamma='scale', C = 10),
LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)
]

data = [
[dataset.boston_x, dataset.boston50_y],
[dataset.boston_x, dataset.boston75_y],
[dataset.digit_x, dataset.digit_y]
]

model_name = ["LinearSVC", "SVC", "LogisticRegression"]

data_name = ["Boston50", "Boston75", "Digits"]

# loop through datasets and methods to get results for every module
def q3ii():
    for i in range(0, 3):
        for j in range(0, 3):
            print("Error rates for", model_name[i], "with", data_name[j])
            my_train_test.my_train_test(method[i], data[j][0], data[j][1],0.75, 10)


if __name__ == "__main__":
    q3ii()
