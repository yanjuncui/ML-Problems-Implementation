# Load and dispose datasets
# all data variables are set as global variables to make them easier being used in main functions


from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
import numpy as np

global boston_x, boston50_y, boston75_y, digit_x, digit_y
boston = load_boston()
boston_x, y = boston.data, boston.target
mid50 = np.percentile(y, 50)
# print(mid50)
boston50_y = np.where(y>mid50, 1, 0)
# print(boston50_y)
mid75 = np.percentile(y, 75)
# print(mid75)
boston75_y =np.where(y>mid75, 1, 0)
# print(boston75_y)

digits = load_digits()
digit_x, digit_y = digits.data, digits.target
