import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from skimage import io

pic = io.imread('ex7-kmeans and PCA/dataset/bird_small.png') / 255.
io.imshow(pic)

pic.shape

# serialize data
data = pic.reshape(128*128, 3)
def combine_data_C(data, C):
    data_with_c = data.copy()
    data_with_c['C'] = C
    return data_with_c


# k-means fn --------------------------------
def random_init(data, k):
    """choose k sample from data set as init centroids
    Args:
        data: DataFrame
        k: int
    Returns:
        k samples: ndarray
    """
    return data.sample(k).as_matrix()

# 输入：一组(x,y)，聚类中心
# 输出：与该组数据的二范数最小的聚类中心的类号
def _find_your_cluster(x, centroids):
    """find the right cluster for x with respect to shortest distance
    Args:
        x: ndarray (n, ) -> n features
        centroids: ndarray (k, n)
    Returns:
        k: int
        """
    # apply_along_axis(func,axis,arr,*args,*kwargs)
    # 沿轴返回调用func函数的arr值，其中axis=1为沿x轴
    distances = np.apply_along_axis(func1d=np.linalg.norm,  # this give you l2 norm
                                    axis=1,
                                    arr=centroids - x)  # use ndarray's broadcast
    # 返回最小值的下标，也就是返回某点(x,y)与聚类中心k的最小范数.
    # 在这个例子中，聚类中心有三个，分别为centroids。
    return np.argmin(distances)

# 输入：带分类数据，聚类中心
# 输出：每个数据的类号

def assign_cluster(data, centroids):
    """assign cluster for each node in data
    return C ndarray
    """
    # 将data化为数组，让其每组值都进行调用_fing_yout_cluster()一次，以获得每组与哪个簇的二范数最小。
    return np.apply_along_axis(lambda x: _find_your_cluster(x, centroids),
                               axis=1,
                               arr=data.as_matrix())


# 通过第一次的分类，得出第一次分类结果。
# 也就可以通过第一次的分类结果，得出第一次分类中三个簇的(x,y)均值。
# 输入：为带分类值和上一次的分类结果。
# 输出：为根据上一次分类结果得出的各个簇的(x,y)均值，且输出的顺序为按C的类号升序排列
def new_centroids(data, C):
    data_with_c = combine_data_C(data, C)

    return data_with_c.groupby('C', as_index=False).\
                       mean().\
                       sort_values(by='C').\
                       drop('C', axis=1).\
                       as_matrix()

# 输入 待分类数据，聚类中心，类别号
# 输出分类损失值
def cost(data, centroids, C):
    m = data.shape[0]

    expand_C_with_centroids = centroids[C]

    distances = np.apply_along_axis(func1d=np.linalg.norm,
                                    axis=1,
                                    arr=data.as_matrix() - expand_C_with_centroids)
    return distances.sum() / m

#输入：待分类原始数值,簇个数，迭代次数，停止迭代阈值
#输出：迭代训练后的C值，迭代训练后的聚类中心，达到阈值时的损失值
def _k_means_iter(data, k, epoch=100, tol=0.0001):
    """one shot k-means
    with early break
    """
    centroids = random_init(data, k)
    cost_progress = []

    for i in range(epoch):
        print('running epoch {}'.format(i))
        # 得到分类号
        C = assign_cluster(data, centroids)
        # 得到新的聚类中心
        centroids = new_centroids(data, C)
        # 损失值
        cost_progress.append(cost(data, centroids, C))

        if len(cost_progress) > 1:  # early break
            if (np.abs(cost_progress[-1] - cost_progress[-2])) / cost_progress[-1] < tol:
                break

    return C, centroids, cost_progress[-1]

# 输入:待分类原始数据,类别数,迭代次数,迭代的轮数
# 输出:迭代后,最小损失值的类别号,聚类中心及损失值
def k_means(data, k, epoch=100, n_init=10):
    """do multiple random init and pick the best one to return
    Args:
        data (pd.DataFrame)
    Returns:
        (C, centroids, least_cost)
    """

    tries = np.array([_k_means_iter(data, k, epoch) for _ in range(n_init)])

    least_cost_idx = np.argmin(tries[:, -1])

    return tries[least_cost_idx]

C, centroids, cost = k_means(pd.DataFrame(data), 16, epoch = 10, n_init=3)

# 调用sklearn包模式

from sklearn.cluster import KMeans

model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
model.fit(data)
# 图片降为/压缩后用 一个(16,3)和(16384,1)的数组就可以近似表示了。
centroids = model.cluster_centers_
print(centroids.shape)
C = model.predict(data)
print(C.shape)
centroids[C].shape
compressed_pic = centroids[C].reshape((128,128,3))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()



