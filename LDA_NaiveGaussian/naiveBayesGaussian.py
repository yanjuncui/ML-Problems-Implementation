import dataset
import my_cross_val
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

data = [
[dataset.boston_x, dataset.boston50_y],
[dataset.boston_x, dataset.boston75_y],
[dataset.digit_x, dataset.digit_y]
]

class NBG1d:
    def fit(self, X, y):
        # print(y)
        self.k = len(np.unique(y))
        seperate_X = []
        for i in range(self.k):
            seperate_X.append([])
        for i in range(len(X)):
            # print(int(y[i]))
            seperate_X[int(y[i])].append(X[i])
        self.seperate_X = seperate_X

        pc = []
        for i in range(self.k):
            pc.append(len(seperate_X[i])/len(X))
        self.pc = pc

        seperate_mean = []
        for i in range(self.k):
            seperate_mean.append(np.mean(seperate_X[i], axis = 0))
        self.seperate_mean = seperate_mean

        seperate_var = []
        for i in range(self.k):
            seperate_var.append(np.var(seperate_X[i], axis = 0))
        self.seperate_var = seperate_var


    def predict(self, X):
        # print(X[0])
        y_pred = []
        for i in range(len(X)):
            gaussian = []
            if self.k == 2:
                score = []
                for j in range(self.k):
                    gaussian.append((np.exp(-(X[i]-self.seperate_mean[j])**2/(2*self.seperate_var[j])))*(1/np.sqrt(2*np.pi)*self.seperate_var[j]))
                for j in range(self.k):
                    score.append(np.sum(np.log(gaussian[j]))+np.log(self.pc[j]))
                score = np.array(score)
            else:
                score = [0]*self.k
                for j in range(self.k):
                    score[j] = np.log(self.pc[j])
                    for m in range(len(X[0])):
                        # print(score[-1])
                        # print(self.pc[j])
                        score[j] += self.pc[j]*np.log(self.pc[j])

                    # print(np.argmax(score))
            y_pred.append(np.argmax(score))
        # print(y_pred)
        return y_pred

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
    for i in range(0, N):
        classNumList[int(y[i])]+=1
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


# get diagonal covarience for one class
def getDiagCov(cov_matrix):
    N, D = np.shape(cov_matrix)
    diag_matrix = np.asarray([[0.0]*D]*N)
    for i in range(0, N):
        diag_matrix[i][i] = cov_matrix[i][i]
    # print(diag_matrix)
    return diag_matrix

# get diagonal covarience for all classes
def getDiagCovList(X, y, k, d):
    covList = getCovList(X, y, k, d)
    diagList = []
    for i in range(0, k):
        diagList.append(getDiagCov(covList[i]))
    return diagList

class NBG2d:

    # =======================ALL are helper functions above==================
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
        self.covs = getDiagCovList(X, y, self.k, self.d)
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


def naiveBayesGaussian(num_splits, train_percent):
    method1 = NBG1d()
    for i in range(2):
        if i==0:
            print("Boston50 dataset: ")
        elif i==1:
            print("==========")
            print("boston75 dataset: ")
        my_cross_val.my_cross_val_q4(method1, data[i][0],data[i][1], num_splits, train_percent)
    method2 = NBG2d(k = len(np.unique(dataset.digit_y)),  d = np.shape(dataset.digit_x)[1])
    print("==========")
    print("digit dataset: ")
    my_cross_val.my_cross_val_q4(method2, data[2][0],data[2][1], num_splits, train_percent)


if __name__ == "__main__":
    naiveBayesGaussian(10, [10, 25, 50, 75, 100])
