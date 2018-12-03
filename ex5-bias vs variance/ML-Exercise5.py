import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """for ex5
    d['X'] shape = (12, 1)
    pandas has trouble taking this 2d ndarray to construct a dataframe, so I ravel
    the results
    """
    d = sio.loadmat('dataset/ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])

X, y, Xval, yval, Xtest, ytest = load_data()

df = pd.DataFrame({'water_level':X, 'flow':y})
# (x轴，y轴，数据)
sns.lmplot('water_level', 'flow', data=df, fit_reg=False, size=7)
plt.show()

# insert(待插入的数据，插入的位置，准备插入的数据，沿什么轴插入)
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]

def cost(theta, X, y):
    """
    X: R(m*n), m records, n features
    y: R(m)
    theta : R(n), linear regression parameters
    """
    m = X.shape[0]

    inner = X @ theta - y  # R(m*1)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost

theta = np.ones(X.shape[1])
cost(theta, X, y)

def gradient(theta, X, y):
    m = X.shape[0]

    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)

    return inner / m

gradient(theta, X, y)


def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = theta.copy()  # same shape as theta
    regularized_term[0] = 0  # don't regularize intercept theta

    regularized_term = (l / m) * regularized_term

    return gradient(theta, X, y) + regularized_term

regularized_gradient(theta, X, y)


# 该函数用于求解训练后的theta值
def linear_regression_np(X, y, l=1):
    """linear regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.ones(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    return res

def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()

    return cost(theta, X, y) + regularized_term

theta = np.ones(X.shape[0])

final_theta = linear_regression_np(X, y, l=0).get('x')

b = final_theta[0] # intercept
m = final_theta[1] # slope

# 可视化训练数据
plt.scatter(X[:,1], y, label="Training data")
# 可视化代入求解出theta后假设函数
plt.plot(X[:, 1], X[:, 1]*m + b, label="Prediction")
# label的位置 loc = 2在左上角
plt.legend(loc=2)
plt.show()

training_cost, cv_cost = [], []

m = X.shape[0]

# 得出 训练集和交叉测试集的regularized_cost(res.x, X[:i, :], y[:i], l=0) for i in X.shape=[0]值

for i in range(1, m + 1):
    #     print('i={}'.format(i))
    res = linear_regression_np(X[:i, :], y[:i], l=0)

    tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
    cv = regularized_cost(res.x, Xval, yval, l=0)
    #     print('tc={}, cv={}'.format(tc, cv))
    training_cost.append(tc)
    cv_cost.append(cv)
# 可视化测试集的损失值(x轴为 i in X.shape[0])
plt.plot(np.arange(1, m+1), training_cost, label='training cost')
# 得出交叉测试集的损失值(x轴为 i in X.shape[0])
plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
plt.legend(loc=1)
plt.show()

def prepare_poly_data(*args, power):
    """
    args: keep feeding in X, Xval, or Xtest
        will return in the same order
    """
    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)

        # normalization
        ndarr = normalize_feature(df).as_matrix()

        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]

#创建多项式特征
# 给其进行元素乘方
def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.as_matrix() if as_ndarray else df

X, y, Xval, yval, Xtest, ytest = load_data()

poly_features(X, power=3)

# 准备多项式回归数据

#对数据进行按列标准化 按列的形式
def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())

X_poly, Xval_poly, Xtest_poly= prepare_poly_data(X, Xval, Xtest, power=8)
X_poly[:3, :]

# 画出学习曲线
def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression_np(X[:i, :], y[:i], l=l)

        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)
# 得到l = 0 时 训练集 和交叉测试在多项回归后的损失值
plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
plt.show()

# 得到l = 1时 训练集 和交叉测试在多项回归后的损失值
plot_learning_curve(X_poly, y, Xval_poly, yval, l=1)
plt.show()

# 得到l = 100 时 训练集 和交叉测试在多项回归后的损失值
plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)
plt.show()

#取一系列的l 值，找出其中使得交叉测试集损失值最小的

l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []

for l in l_candidate:
    res = linear_regression_np(X_poly, y, l)

    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)

    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(l_candidate, training_cost, label='training')
plt.plot(l_candidate, cv_cost, label='cross validation')
plt.legend(loc=2)

plt.xlabel('lambda')

plt.ylabel('cost')
plt.show()

# best cv I got from all those candidates
l_candidate[np.argmin(cv_cost)]

# use test data to compute the cost
# 找出上述l值中使得测试值最小的
for l in l_candidate:
    theta = linear_regression_np(X_poly, y, l).x
    print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))

# 调参后， $\lambda = 0.3$ 是最优选择，这个时候测试代价最小
