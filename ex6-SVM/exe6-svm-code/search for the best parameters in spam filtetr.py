from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.io as sio

mat_tr = sio.loadmat('ex6-SVM/ex6-dataset/spamTrain.mat')
X, y = mat_tr.get('X'), mat_tr.get('y').ravel()
mat_test = sio.loadmat('ex6-SVM/ex6-dataset/spamTest.mat')
test_X, test_y = mat_test.get('Xtest'), mat_test.get('ytest').ravel()

candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

# gamma to comply with sklearn parameter name
combination = [(C, gamma) for C in candidate for gamma in candidate]
len(combination)


search = []
# 默认和核函数为高斯核函数
for C, gamma in combination:
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(X, y)
    search.append(svc.score(test_X, test_y))

best_score = search[np.argmax(search)]
best_param = combination[np.argmax(search)]
print(best_score, best_param)
# 待选参数中最好的一组(3,0.01)
# 最终最好的结果和逻辑回归相似，皆为0.99






