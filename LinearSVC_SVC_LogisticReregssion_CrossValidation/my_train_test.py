import numpy as np
# get how many data should be in train set
def parts (pi, dataset):
    train_num = round(pi*len(dataset))
    return train_num

# split data, get train and test dataset
def split_data (pi, dataset_x, dataset_y):
    dataset_x, dataset_y = randomize_order (dataset_x, dataset_y)
    train_num = parts(pi, dataset_x)

    train_x = dataset_x[ :train_num]
    test_x = dataset_x[train_num: ]

    train_y = dataset_y[ :train_num]
    test_y = dataset_y[train_num: ]

    return train_x, train_y, test_x, test_y

# randomrize datasets' order
def randomize_order (x, y):
    sum_data = []
    for i in range(0, len(y)):
        sum_data.append([x[i], y[i]])
    # print(sum_data)
    np.random.shuffle(sum_data)
    # print(sum_data)
    x = []
    y = []
    for i in range(0, len(sum_data)):
        x.append(sum_data[i][0])
        y.append(sum_data[i][1])
    # print(x)
    # print(y)
    return x, y

# after train one time, conbine dataset together
def concatenate_data(dataset_x, dataset_y, test_x, test_y):
    np.append(test_x, dataset_x)
    np.append(test_y, dataset_y)
    return dataset_x, dataset_y

# train dataset k times
def my_train_test(method, X, y, pi, k):
    res = []
    for i in range(0, k):
        train_x, train_y, test_x, test_y = split_data(pi, X, y)
        clf = method.fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        error = 1- np.sum(pred_y==test_y)/len(test_x)
        res.append(error)
        print("Fold ", i+1,  ": ", error)
        concatenate_data(X, y, test_x, test_y)
    print("Mean:", np.mean(res))
    print("Standard deviation:", np.std(res))
