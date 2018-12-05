# 支持向量机部分学习总结

## 支持向量机简述 

    支持向量机是监督学习算法中的一种，其主要是用在分类任务中，典型的使用场景有垃圾邮件的分类。
  
## 本章学习重点

* 理解支持向量机的数学原理及其和逻辑回归的不同之处。

* 理解支持向量机中核函数的数学原理，及其作用。

* 代码实现支持向量机。

---

### 一、 支持向量机的数学原理
   
    吴恩达视频中在讨论时总是假设我们的假设函数是一个s函数，需要注意。
    
#### 1.1 支持向量机的损失函数（周志华和李航书给的都很复杂，这里只给吴恩达教学视频中的）    

        
![支持向量机的损失函数](https://raw.githubusercontent.com/pengxl8518/machine_learning_Andrew_Ng/master/ex6-SVM/svm_theory/svm%E5%81%87%E8%AE%BE%E5%87%BD%E6%95%B0-%E5%90%B4%E6%81%A9%E8%BE%BE.png)
   
   * 根据吴恩达所言，把损失函数从A+r×B的优化问题变成 C×A+B的优化问题只是因为是个惯例。
    
   * 在A+r×B的问题中，如果给定r为一个很大的值则意味着给B一个很大的权重。
   
   * 在SVM的损失函数中：
   
   * C较大时，相当于r较小可能会导致过拟合，高方差。
   
   * C较小时，相当于r较大可能会导致低拟合，高偏差。
   
   
#### 1.2 支持向量机的可视化边界理解

   * 下图中，theat × X>1时 y=1时建立在假设函数为s函数之上的。包括下面对最大间距的解释上也是。

![SVM的决策边界的可视化理解1](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C.png)

   * 巧妙的将变量x与theta之间的关系变成了两个范数值(p投影与theta)的相乘。
    
![SVM的决策边界2](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C2.png)

   * 过一个简单二分类问题来解释x×theta的等价表达式pi成|theta|的反比例关系。
    
![SVM的决策边界3，重点理解](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C3(%E9%87%8D%E7%82%B9%E7%90%86%E8%A7%A3).png)
    
   * 接上
    
![SVM的决策边界4，重点理解](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C4(%E9%87%8D%E7%82%B9%E7%90%86%E8%A7%A32).png)
    
   * 接上
    
![SVM的决策边界5，重点理解](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C5(%E9%87%8D%E7%82%B9%E7%90%86%E8%A7%A33).png)
  
  * 以上就是为什么支持向量机最终会找到最大间距分类器的原因。因为它试图极小化theta的范数值来 极大化p(i)的范数。
  
  * **在这个例子中，theta向量与超平面成90°角，所以x在theta上的投影就是x与超平面的距离。**

### 二、 支持向量机的与逻辑回归的不同    

![图片](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%92%8C%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%9A%84%E4%B8%80%E8%88%AC%E6%80%A7%E5%8C%BA%E5%88%AB.png)
    
    
    
 #### 三、支持向量机核函数的原理
 
 
   * 在高级多项式模型问题无法用直线进行分隔的分类中，我们往往需要更加复杂的假设函数去拟合数据。而核函数则可以帮助我们去构造一些更复杂的假设函数,他的原理是利用核函数来选取新的特征。
    
   * 不使用核函数数的支持向量机，它的核叫做线性核函数。
    
   * 一般而言，C/o 对模型的影响如下。
    
    
    
    
#### 四、SVM中普遍使用的准则





   * 好的算法很重要，但更重要的是你有多少数据，你是否熟练和擅长做误差分析和排除学习算法，是否有能力设定新的特征变量和找出其他能觉决定你学习算法的变量。
    
    
    
#### 五、 代码学习

   * 实际解决问题时，通常不自己去编写求解theta的函数，一般都是建议调用别人写好的库。如sklearn库。
    
   *  调用别人的库很简单很方便，但不利于学习SVM算法的更多底层细节。
    
   * 在垃圾邮件分类中，逻辑回归比SVM得到了更好的结果。而SVM模型则在我通过对一系列值进行寻找后才得到了和逻辑回归简单训练的结果

    
    
    
    
    
    
    
 
 
    
 
    
    
    
    
    
    

  
