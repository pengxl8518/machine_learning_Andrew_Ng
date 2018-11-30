# 吴恩达机器学习笔记--神经网络

* 1：核心内容

   前馈神经网络、bp神经网络的公式推导。

* 2：学习时的困惑

   误把bp神经网络中的误差理解成，后一层与前一层的值之差。但实际上，在吴恩达视频中的‘误差’，实际上他指的是一个微分值。详情见[Machine Learning |吴恩达 （2-2）---神经网络,反向传播推导（超简单版）](https://blog.csdn.net/weixin_40920228/article/details/80709216)

* 3：代码实现的逻辑
   
   1、加载可视化元素像素数据
   
   2、将提供好的theta值代入前馈网络中，并求解网络中每个神经元的值
   
   3、初始化theata值在(low,hight)之间，定义损失函数、带正则化的损失函数，并调用别人写好的函数求解出theta值。
   
   4、评估准确度
   
   5、可视化输出结果。似乎并不理想。
   
   
* 4：代码中觉得写的很不错的

   1、降维函数

   ```
   # ravel 用于降维，默认降维的顺序为行序优先。与其用相似的还用flatten
   def serialize(a, b):
       return np.concatenate((np.ravel(a), np.ravel(b)))
   ```

   2、转化维度
   ```
   def deserialize(seq):
   #     """into ndarray of (25, 401), (10, 26)"""
       return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)
   ```

   3、初始化theta值函数

   ```
   def random_init(size):
       return np.random.uniform(-0.12, 0.12, size)
   
   ```
 
   4、编码函数（上一章的代码）

   ```
   #将y值从1-10的真实值，映射成一个10×5000维的矩阵，每一行代表一个值的0/1映射转换
   for k in range(1, 11):
      y_matrix.append((raw_y == k).astype(int))

   ```


  ![输出结果可视化](https://raw.githubusercontent.com/pengxl8518/machine-learning-/master/figure_1.png)输出结果可视化
  
  
  [markdown使用技巧]（https://www.jianshu.com/p/38fe4911b4a0）
