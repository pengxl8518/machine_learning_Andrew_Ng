from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import scipy.io as sio

mat_tr = sio.loadmat('ex6-SVM/ex6-dataset/spamTrain.mat')
mat_tr.keys()
X, y = mat_tr.get('X'), mat_tr.get('y').ravel()
X.shape, y.shape

mat_test = sio.loadmat('ex6-SVM/ex6-dataset/spamTest.mat')
mat_test.keys()
test_X, test_y = mat_test.get('Xtest'), mat_test.get('ytest').ravel()
test_X.shape, test_y.shape

# fit SVM model
svc = svm.SVC()
svc.fit(X, y)
pred = svc.predict(test_X)
print(metrics.classification_report(test_y, pred))

# fit linear logistic regresion
logit = LogisticRegression()
logit.fit(X, y)
pred = logit.predict(test_X)
print(metrics.classification_report(test_y, pred))