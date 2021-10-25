import numpy as np

# split datasets, extract one part as test data
# and combine other data sets to be train data
def split_data (dataset_x, dataset_y, k, i):
    dataset_x = np.array_split(dataset_x, k)

    test_x = dataset_x.pop(i)
    train_x = np.concatenate(dataset_x)

    dataset_y = np.array_split(dataset_y, k)

    test_y = dataset_y.pop(i)
    train_y = np.concatenate(dataset_y)
    return train_x, train_y, test_x, test_y

# after train once, combine data together to start the next train
def concatenate_data(dataset_x, dataset_y, test_x, test_y):
    np.append(test_x, dataset_x)
    np.append(test_y, dataset_y)
    return dataset_x, dataset_y

# repeat training k times, and calculate the error
def my_cross_val(method, X, y, k):
    res = []
    for i in range(0, k):
        train_x, train_y, test_x, test_y = split_data(X, y, k, i)
        method.fit(train_x, train_y)
        pred_y = method.predict(test_x)
        error = 1- np.sum(pred_y==test_y)/len(test_x)
        res.append(error)
        print("Fold ", i+1,  ": ", error)
        concatenate_data(X, y, test_x, test_y)
    print("Mean:", np.mean(res))
    print("Standard deviation:", np.std(res))
