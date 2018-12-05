# 支持向量机部分学习总结

## 支持向量机简述 

    支持向量机是监督学习算法中的一种，其主要是用在分类任务中，典型的使用场景有垃圾邮件的分类。
  
## 本章学习重点

* 理解支持向量机的数学原理及其和逻辑回归的不同之处。

* 理解支持向量机中核函数的数学原理，及其作用。

* 代码实现支持向量机。

### 支持向量机的数学原理
    
* 支持向量机的假设函数（周志华和李航书给的都很复杂，这里只给吴恩达教学视频中的）    
        
![图片](https://raw.githubusercontent.com/pengxl8518/machine_learning_Andrew_Ng/master/ex6-SVM/svm_theory/svm%E5%81%87%E8%AE%BE%E5%87%BD%E6%95%B0-%E5%90%B4%E6%81%A9%E8%BE%BE.png)

![图片](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C.png)

![图片](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C2.png)

![图片](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C3(%E9%87%8D%E7%82%B9%E7%90%86%E8%A7%A3).png)

![图片](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C4(%E9%87%8D%E7%82%B9%E7%90%86%E8%A7%A32).png)

![图片](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C5(%E9%87%8D%E7%82%B9%E7%90%86%E8%A7%A33).png)

![图片](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex6-SVM/svm_theory/svm%E5%92%8C%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%9A%84%E4%B8%80%E8%88%AC%E6%80%A7%E5%8C%BA%E5%88%AB.png)
    
    
    
    
    
  以上就是为什么支持向量机最终会找到最大间距分类器的原因。因为它试图极小化theta的范数值来
        极大化p(i)的范数。
    
    
    
    
    
    

  
