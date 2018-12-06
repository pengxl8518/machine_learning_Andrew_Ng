import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio
# data1
mat = sio.loadmat('ex7-kmeans and PCA/dataset/ex7data1.mat')
mat.keys()
data1 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])

sns.set(context="notebook", style="white")

sns.lmplot('X1', 'X2', data=data1, fit_reg=False)
plt.show()

# data2
mat = sio.loadmat('ex7-kmeans and PCA/dataset/ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data2.head()

sns.lmplot('X1', 'X2', data=data2, fit_reg=False)
plt.show()


