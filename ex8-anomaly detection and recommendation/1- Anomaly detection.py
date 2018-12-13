import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white", palette="RdBu")

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.cross_validation import train_test_split

mat = sio.loadmat('ex8-anomaly detection and recommendation/dataset/ex8data1.mat')
mat.keys()
X = mat.get('X')

Xval, Xtest, yval, ytest = train_test_split(mat.get('Xval'),
                                            mat.get('yval').ravel(),
                                            test_size=0.5)

# scatter_kws:要传递给plt.scatter和plt.plot的其他关键字参数。
# s表示标记点的大小，默认为point*2。注：点越大越模糊。
# alpha  让显示的标记点置于【透明:0，不透明:1】之间。
sns.regplot('Latency', 'Throughput',
           data=pd.DataFrame(X, columns=['Latency', 'Throughput']),
           fit_reg=False,
           scatter_kws={"s":20,
                        "alpha":0.5})
plt.show()

mu = X.mean(axis=0)
print(mu, '\n')
# 求协方差
cov = np.cov(X.T)
print(cov)


# example of creating 2d grid to calculate probability density
np.dstack(np.mgrid[0:3,0:3])

# create multi-var Gaussian model
# 求多元正态随机变量
multi_normal = stats.multivariate_normal(mu, cov)

# create a grid
x, y = np.mgrid[0:30:0.01, 0:30:0.01]
pos = np.dstack((x, y))

fig, ax = plt.subplots()

# plot probability density
#pdf(x,*args,**kwds) 给定RV的x处的概率密度函数
# ax.contourf用于绘制轮廓
ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')

# plot original data points
#ax使用ax=ax轴，意味着对象将会使用当前轴(ax)进行绘图 ???。
sns.regplot('Latency', 'Throughput',
           data=pd.DataFrame(X, columns=['Latency', 'Throughput']),
           fit_reg=False,
           ax=ax,
           scatter_kws={"s":10,
                        "alpha":0.4})
plt.show()

# select threshold $\epsilon$
# 这个函数里是以f1函数作为衡量指标。
def select_threshold(X, Xval, yval):
    """use CV data to find the best epsilon
    Returns:
        e: best epsilon with the highest f-score
        f-score: such best f-score
    """
    # create multivariate model using training data
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    # this is key, use CV data for fine tuning hyper parameters
    # 这个pval的值是Xval对应的多元高斯值？
    pval = multi_normal.pdf(Xval)

    # set up epsilon candidates
    # linspace(start,stop,num) 把开始-终止之间的值切成num个
    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)

    # calculate f-score
    #小于的e会被标记为1，即为异常。
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')
        fs.append(f1_score(yval, y_pred))

    # find the best f-score
    argmax_fs = np.argmax(fs)
    # 返回最大的f1值对应的pval值，及fs值
    return epsilon[argmax_fs], fs[argmax_fs]

from sklearn.metrics import f1_score, classification_report

e, fs = select_threshold(X, Xval, yval)
print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))

# visualize prediction of Xval using learned $\epsilon$
def select_threshold(X, Xval, yval):
    """use CV data to find the best epsilon
    Returns:
        e: best epsilon with the highest f-score
        f-score: such best f-score
    """
    # create multivariate model using training data
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    # this is key, use CV data for fine tuning hyper parameters
    pval = multi_normal.pdf(Xval)

    # set up epsilon candidates
    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)

    # calculate f-score
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')
        fs.append(f1_score(yval, y_pred))

    # find the best f-score
    argmax_fs = np.argmax(fs)

    return epsilon[argmax_fs], fs[argmax_fs]


def predict(X, Xval, e, Xtest, ytest):
    """with optimal epsilon, combine X, Xval and predict Xtest
    Returns:
        multi_normal: multivariate normal model
        y_pred: prediction of test data
    """
    Xdata = np.concatenate((X, Xval), axis=0)

    mu = Xdata.mean(axis=0)
    cov = np.cov(Xdata.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    # calculate probability of test data
    pval = multi_normal.pdf(Xtest)
    y_pred = (pval <= e).astype('int')

    print(classification_report(ytest, y_pred))
    # 返回预测的y值（异常与否）
    return multi_normal, y_pred

multi_normal, y_pred = predict(X, Xval, e, Xtest, ytest)



# construct test DataFrame
data = pd.DataFrame(Xtest, columns=['Latency', 'Throughput'])
# 新增预测数据y_pred
data['y_pred'] = y_pred

# create a grid for graphing
x, y = np.mgrid[0:30:0.01, 0:30:0.01]
pos = np.dstack((x, y))

fig, ax = plt.subplots()

# plot probability density
ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')

# plot original Xval points
sns.regplot('Latency', 'Throughput',
            data=data,
            fit_reg=False,
            ax=ax,
            scatter_kws={"s":10,
                         "alpha":0.4})

# mark the predicted anamoly of CV data. We should have a test set for this...
# 将测试集数据中的异常数据标记出来
anamoly_data = data[data['y_pred']==1]
ax.scatter(anamoly_data['Latency'], anamoly_data['Throughput'], marker='x', s=50)
plt.show()

# high dimension data

mat = sio.loadmat('./data/ex8data2.mat')

X = mat.get('X')
Xval, Xtest, yval, ytest = train_test_split(mat.get('Xval'),
                                            mat.get('yval').ravel(),
                                            test_size=0.5)
e, fs = select_threshold(X, Xval, yval)
print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))

multi_normal, y_pred = predict(X, Xval, e, Xtest, ytest)

print('find {} anamolies'.format(y_pred.sum()))

mat = sio.loadmat('ex8-anomaly detection and recommendation/dataset/ex8data2.mat')
X = mat.get('X')
Xval, Xtest, yval, ytest = train_test_split(mat.get('Xval'),
                                            mat.get('yval').ravel(),
                                            test_size=0.5)
