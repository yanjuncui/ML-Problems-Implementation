from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import dataset
import rand_proj
import quad_proj
import my_cross_val
import warnings
warnings.filterwarnings("ignore")

method = [
LinearSVC(max_iter = 2000),
SVC(gamma='scale', C = 10),
LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)
]

X1 = rand_proj.rand_proj (dataset.digit_x, 32)
X2 = quad_proj.quad_proj (dataset.digit_x)


model_name = ["LinearSVC", "SVC", "LogisticRegression"]
data_name = ["X1", "X2"]
dataset = [X1, X2, dataset.digit_y]

# loop through datasets and methods to get results for every module
def q4():
    for i in range(0, 3):
        for j in range(0,2):
            print("Error rates for", model_name[i], "with ", data_name[j])
            my_cross_val.my_cross_val(method[i], dataset[j], dataset[2], 10)

if __name__ == "__main__":
    q4()
