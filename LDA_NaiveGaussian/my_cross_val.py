import numpy as np

# split datasets, extract one part as test data
# and combine other data sets to be train data
def split_data (dataset_x, dataset_y, i):
    dataset = np.append(dataset_x, dataset_y.reshape(len(dataset_y), 1), axis = 1)
    np.random.shuffle(dataset)
    dataset_y = dataset[:, len(dataset[0])-1]
    dataset_x = np.delete(dataset, -1, axis=1)

    dataset_x = np.array_split(dataset_x, 5)

    test_x = dataset_x.pop(0)
    train_x = np.concatenate(dataset_x)

    dataset_y = np.array_split(dataset_y, 5)

    test_y = dataset_y.pop(0)
    train_y = np.concatenate(dataset_y)
    return train_x, train_y, test_x, test_y

# after train once, combine data together to start the next train
def concatenate_data(dataset_x, dataset_y, test_x, test_y):
    np.append(test_x, dataset_x)
    np.append(test_y, dataset_y)
    return dataset_x, dataset_y

# repeat training k times, and calculate the error
def my_cross_val_q31(method, X, y, k):
    res = []
    for i in range(0, k):
        train_x, train_y, test_x, test_y = split_data(X, y, i)
        w, w0 = method.fit(train_x, train_y)
        pred_y = method.predict(test_x, w, w0)
        error = 1- np.sum(pred_y==test_y)/len(test_x)
        res.append(error)
        print("Fold ", i+1,  ": ", error)
        concatenate_data(X, y, test_x, test_y)
    print("Mean:", np.mean(res))
    print("Standard deviation:", np.std(res))

def my_cross_val_q32(method, X, y, k):
    res = []
    for i in range(0, k):
        train_x, train_y, test_x, test_y = split_data(X, y, i)
        method.fit(train_x, train_y)
        pred_y = method.predict(test_x)
        error = 1- np.sum(pred_y==test_y)/len(test_x)
        res.append(error)
        print("Fold ", i+1,  ": ", error)
        concatenate_data(X, y, test_x, test_y)
    print("Mean:", np.mean(res))
    print("Standard deviation:", np.std(res))


def my_cross_val_q4(method, X, y, num_splits, train_percent):
    res = np.zeros((len(train_percent), num_splits))
    for i in range(0, num_splits):
        print("-----")
        train_x, train_y, test_x, test_y = split_data(X, y, i)
        for j in range(len(train_percent)):
            train_num = int(len(train_x) * 0.01 * train_percent[j])
            part_train_x = train_x[:train_num]
            part_train_y = train_y[:train_num]
            method.fit(part_train_x, part_train_y)
            pred_y = method.predict(test_x)
            error = 1- np.sum(pred_y==test_y)/len(test_x)
            res[j][i] = error
            print("Fold ", i+1, "train percent ", train_percent[j], ": ", error)
            concatenate_data(X, y, test_x, test_y)
    for j in range(len(train_percent)):
        print("Mean for percent:",train_percent[j], "is", np.mean(res[j]))
        print("Standard deviation for percent:", train_percent[j], "is", np.std(res[j]))
