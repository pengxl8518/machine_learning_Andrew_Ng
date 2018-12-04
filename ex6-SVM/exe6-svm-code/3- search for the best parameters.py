from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('ex6-SVM/ex6-dataset/ex6data3.mat')
print(mat.keys())

training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
training['y'] = mat.get('y')

cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
cv['y'] = mat.get('yval')

candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

# gamma to comply with sklearn parameter name
combination = [(C, gamma) for C in candidate for gamma in candidate]
len(combination)


search = []
# 默认和核函数为高斯核函数
for C, gamma in combination:
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(training[['X1', 'X2']], training['y'])
    search.append(svc.score(cv[['X1', 'X2']], cv['y']))

best_score = search[np.argmax(search)]
best_param = combination[np.argmax(search)]

print(best_score, best_param)

best_svc = svm.SVC(C=100, gamma=0.3)
best_svc.fit(training[['X1', 'X2']], training['y'])
ypred = best_svc.predict(cv[['X1', 'X2']])

print(metrics.classification_report(cv['y'], ypred))


# 方法二：sklearn GridSearchCV

# 候选参数，每个参数的格式需为list形
parameters = {'C': candidate, 'gamma': candidate}
svc = svm.SVC()
# n_jobs为并行作业数，n_jobs=-1代表使用所用处理器
clf = GridSearchCV(svc, parameters, n_jobs=-1)
clf.fit(training[['X1', 'X2']], training['y'])

clf.best_params_

clf.best_score_

ypred = clf.predict(cv[['X1', 'X2']])
print(metrics.classification_report(cv['y'], ypred))

# 两种方法的候选值相同，
# 得到的最优参数不同的原因是GridSearch法将部分输入数据当做了cv集，
# 并用他来找到最佳候选参数







