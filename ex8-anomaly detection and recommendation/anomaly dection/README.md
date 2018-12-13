# 本章主要学习的是对异常数据进行检测

    学习重点：如何检查出数据源中的异常数据。
    
---

 ## 本章学习重点
 
  * 异常检测的意义
  
  * 异常检测的原理--高斯分布
  
  * 异常检测与监督学习的对比
  
  * 多元高斯分布-协方差如何影响模型
  
---  
  
 
### 异常检测的意义

  * 异常数据会对模型的训练产生误导，会降低训练出模型的拟合度。
  
  * 一般而言，异常检测在直接对数据进行训练后得不到理解结果时使用。
  
  


### 异常检测的原理-高斯分布

  #### 什么是高斯分布？
  
   * 高斯分布也称正态分布，自然界很多的场景下产生的数据都符合高斯分布。
   
   * 如果一个变量符合高斯分布，则它会服从一个概率密度公式。
   
   * 更多见百度百科。
   
   #### 高斯分布的原理
   
  ![高斯分布的公式](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex8-anomaly%20detection%20and%20recommendation/anomaly%20dection/Principle%20of%20Gaussian%20distribution/%E5%9B%BE%E4%B8%80%EF%BC%9A%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83%E7%9A%84%E5%8E%9F%E7%90%86%E5%85%AC%E5%BC%8F.png)
     
   ![高斯分布的公式](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex8-anomaly%20detection%20and%20recommendation/anomaly%20dection/Principle%20of%20Gaussian%20distribution/%E5%9B%BE%E4%BA%8C%EF%BC%9A%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83%E7%9A%84%E5%8E%9F%E7%90%86%E5%85%AC%E5%BC%8F.png)
     
     
  ### 异常检测与监督学习的对比
     
   ![异常检测与监督学习的不同](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex8-anomaly%20detection%20and%20recommendation/anomaly%20dection/Principle%20of%20Gaussian%20distribution/%E5%9B%BE%E4%B8%89%EF%BC%9A%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E4%B8%8E%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%AF%B9%E6%AF%94.png)
     
     
 ### 多元高斯分布-协方差如何影响模型
     
   ![多元高斯分布的数学原理](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex8-anomaly%20detection%20and%20recommendation/anomaly%20dection/Principle%20of%20Gaussian%20distribution/%E5%9B%BE%E5%9B%9B%EF%BC%9A%E5%A4%9A%E5%85%83%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83%E5%8E%9F%E7%90%86.png)
     
   ![多元高斯分布的数学原理](8F%E6%96%B9%E5%B7%AE%E5%A6%82%E4%BD%95%E5%BD%B1%E5%93%8D%E6%A8%A1%E5%9E%8B%EF%BC%88%E9%87%8D%E7%82%B9%E7%90%86%E8%A7%A3%EF%BC%89.png)
   
   ![多元高斯分布的数学原理](https://github.com/pengxl8518/machine_learning_Andrew_Ng/blob/master/ex8-anomaly%20detection%20and%20recommendation/anomaly%20dection/Principle%20of%20Gaussian%20distribution/%E5%9B%BE%E4%BA%94%EF%BC%9A%E5%8D%8F%E6%96%B9%E5%B7%AE%E5%A6%82%E4%BD%95%E5%BD%B1%E5%93%8D%E6%A8%A1%E5%9E%8B%EF%BC%88%E9%87%8D%E7%82%B9%E7%90%86%E8%A7%A3)
   
 
  
  
    
