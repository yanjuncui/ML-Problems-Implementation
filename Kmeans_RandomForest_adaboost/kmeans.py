import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

def fit(img, k):
    h, w, rgb = np.shape(img)
    rnk = np.zeros((h, w, k))
    uk = np.random.random((k,3))
    LossList = []
    for iter in range(20):
        rnk = np.zeros((h, w, k))
        for m in range(h):
            for n in range(w):
                min_index = -1
                min_dis = 1000
                for i in range(k):
                    dis = np.linalg.norm(img[m, n, :] - uk[i])
                    # print(dis)
                    if dis < min_dis:
                        min_dis = dis
                        min_index = i
                        # print(min_index)
                rnk[m, n, min_index] = 1

        cur_loss = 0
        for i in range(k):
            sum_rx = [0,0,0]
            sum_r = 0
            for m in range(h):
                for n in range(w):
                    if rnk[m, n, i] == 1:
                        # print(img[m, n, :])
                        sum_rx += img[m, n, :]
                        sum_r += 1
                        cur_loss += np.linalg.norm(img[m, n, :] - uk[i])**2
            uk[i] = np.array(sum_rx)/float(sum_r)
        LossList.append(cur_loss)
        print('k =', k, 'loss is', cur_loss)
        if iter > 2 and LossList[-2] == LossList[-1] :
            break

    new_img = np.zeros(np.shape(img))
    for m in range(h):
        for n in range(w):
            for i in range(k):
                if rnk[m, n, i] == 1:
                    new_img[m, n] = uk[i]

    # print('----', LossList)
    plt.subplots()
    plt.imshow(new_img, vmin=0, vmax=1)
    plt.title("K-means image with k = %s"%k)
    plt.show()

    plt.plot(LossList)
    plt.title("Loss evaluation with k =%s"%k)
    plt.show()

def kmeans(image):
    K = [3, 5, 7]
    img = cv2.imread(image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.astype('float') / 255.0
    for i in range(len(K)):
        fit(img, K[i])


if __name__ == "__main__":
    kmeans('umn_csci.png')
