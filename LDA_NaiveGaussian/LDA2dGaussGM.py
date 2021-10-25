import dataset
import my_cross_val
import numpy as np

# helper functions
# put elements in X to the relevant classes
def getClass(X, y, k):
    N, D = np.shape(X)
    classList = [[]]
    for i in range(0, k):
        classList.append([])
        np.asarray(classList[i])
    for i in range(0, N):
        classList[int(y[i])].append(X[i])
    return classList

# Get uniform prior -- Pci
def GetPci(X, y, k):
    classNumList = getClassNum(X, y, k)
    N, D = np.shape(X)
    Pci = []
    for i in range(0, k):
        Pci.append(classNumList[i]/N)
    return Pci

# get the number of elements for all class
def getClassNum(X, y, k):
    N, D = np.shape(X)
    classNumList = [0]*k
    # print(y)
    # np.int64(y)
    for i in range(0, N):
        classNumList[int(y[i])] += 1
    return classNumList

#get mean for one class
def getMean(X, d, classNum, ClassElem):
    sumlist = [0]*d
    avelist = [0]*d
    for i in range(0,classNum):
        for j in range(0,d):
            sumlist[j] = sumlist[j]+ClassElem[i][j]
    # print(sumlist)
    for j in range(0, d):
        avelist[j] = np.divide(sumlist[j],classNum)
    avelist = np.asarray(avelist)
    return avelist

#get mean for all classes
def getMeanList(X, y, k, d):
    classNumList = getClassNum(X, y, k)
    classList = getClass(X, y, k)
    meanList = []
    for i in range(0, k):
        meanList.append(getMean(X, d, classNumList[i], classList[i]))
    return meanList

# get covarience for one class
def getCov(X, d, avelist, classNum, classElem):
    ave_matrix = np.asarray([[1]*d]*classNum)
    ave_matrix = ave_matrix*avelist
    # print(np.shape(ave_matrix))
    cov_matrix = (classElem-ave_matrix).transpose().dot(((classElem-ave_matrix)))/(classNum)
    return cov_matrix+np.eye(d)*0.02 # add a small number of matrix to avoid condition of no inverse

#get covarience for all classes
def getCovList(X, y, k, d):
    meanList = getMeanList(X, y, k, d)
    classNumList = getClassNum(X, y, k)
    classList = getClass(X, y, k)
    covList = []
    for i in range(0, k):
        covList.append(getCov(X, d, meanList[i], classNumList[i], classList[i]))
    return covList

# =======================ALL are helper functions above==================

class LDA2d:
    def __init__(self, k, d):
        self.means = []
        self.pcs = []
        self.covs = []
        self.k = k
        self.d = d
        for i in range(0, k):
          self.means.append(np.zeros(d))
          self.pcs.append(1/k)
          self.covs.append(np.eye(d))

    def fit(self, X, y):
        self.means = getMeanList(X, y, self.k, self.d)
        self.covs =  getCovList(X, y, self.k, self.d)
        self.pcs = GetPci(X, y, self.k)

    def predict(self, X):
        discList = [-1000000000.0]*len(X)
        ypred = [0]*len(X)
        for j in range(0, len(X)):
            for i in range(0, self.k):
                disc = - np.log(np.linalg.det(self.covs[i])) + np.log(self.pcs[i])- (1/2)*np.dot(np.dot((np.subtract(X[j], self.means[i]).transpose()),(np.linalg.inv(self.covs[i]))),(np.subtract(X[j], self.means[i])))
                if disc > discList[j]:
                    discList[j] = disc
                    ypred[j] = i
        return ypred


def LDA2DGaussGM(num_crossval):
    my_cross_val.my_cross_val_q32(LDA2d(k = len(np.unique(dataset.digit_y)), d = np.shape(dataset.digit_x)[1]), dataset.digit_x, dataset.digit_y, num_crossval)

if __name__ == "__main__":
    LDA2DGaussGM(10)
