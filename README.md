# 2022吴恩达机器学习笔记

汇总：[2022吴恩达机器学习笔记](./2022吴恩达机器学习笔记.md)

**目录**

___

[第一章-前言与线性回归](./第一章-前言与线性回归.md)

**前言**

**机器学习的定义**

**监督学习**

- 回归问题-regression problem

- 分类问题-classification problem

**无监督学习**

- 两者的区别

**线性回归-linear regression**

- 前言-线性回归的介绍

- Cost Function(代价函数)

- 梯度下降-Gradient descent algorithm

- 算法的实现

- 算法实现

[第二章-多元线性回归与特征工程](./第二章-多元线性回归与特征工程.md)

**多特征向量（入门）-Multiple features**

**多元线性回归-Multiple linear regression**

- 矢量处理-向量运算

- 多元线性回归的梯度下降-Gradient descent for multiple linear regression

**有关特征值与参数调节工程**

- 特征放缩-Feature scaling

- 检验收敛性-Check the convergence

- - 怎么看收敛性

- - 如何选取$\alpha$

- 特征工程-Feature Engineering

**多项式回归-Polynomial Regression**

[第三章-逻辑回归和正则化](./第三章-逻辑回归和正则化.md)

**逻辑回归-Logistic regression**

- 决策边界

- 逻辑回归的代价函数-Cost function for logistic regression

- 逻辑回归的代价函数

**欠拟合与过拟合-underfitting and overfitting**

- 欠拟合与过拟合的介绍

- 线性回归的正则化

**逻辑回归的正则化**

[第四章-神经网络与前向传播](./第四章-神经网络与前向传播.md)

**开篇**

**神经网络-Neural Networks**

- 神经网络介绍

- 需求预测-Demand Prediction

- Example:视觉处理-Face recognition

- 神经网络层-Neural network layer

- 更加复杂的神经网络

**推理-Inferencing**

- 向前传播(手写识别)-forward propagation

- 代码部分-Inferencing in code

- Tensorflow 中的数据

**构造一个神经网络**

- 调用库写函数

- 单层中的前向传播-Forward prop in a single layer

**AGI猜想-Is there a path to AGI**

**向量化求解-矩阵部分**

**神经网络向量化实现**

[第五章-神经网络实现(激活函数应用)](./第五章-神经网络实现(激活函数应用).md)

**训练神经网络**

- 模型代码介绍

- 训练细节

**激活函数**

- 之前讲过的一些激活函数

- 激活函数的选择

- 激活函数的重要性

**多类问题-Multiclass Problem**

- 多类问题介绍

- Softmax激活函数

- softmax与神经网络

- softmax改进代码

**多标签分类问题-Multi-label classification**

**进阶优化算法**

**小拓展**

[第六章-误差分析与模型改进](./第六章-误差分析与模型改进.md)

**前言**

**模型评估-ebaluating model**

- 应用到线性回归

- 应用到分类问题

**模型选择与交叉验证**

- 选择方法与交叉验证

**模型参数选择**

- 诊断方差偏差

- 正则化参数

**误差的分析**

- 搭建性能-Baseline（基线）

- 学习曲线-Learning curves

- 误差与方差的分析（下一步该改进什么）

**神经网络与误差偏差**

**机器学习的迭代循环**

**如何高效的改进模型**

- 误差分析-error analysis

- 数据添加-adding data

- 迁移学习-transfer learing

**构造系统的全流程**

**公平，偏见，伦理-fairness，bias，ethics**

**选修-数据倾斜与优化方法**

- 数据倾斜介绍

- 权衡精度与召回率

[第七章-决策树与随机森林](./第七章-决策树与随机森林.md)

**决策树**

- 猫猫分类案例-可爱捏

- 学习过程-learning process

- 熵的引入

- 选择拆分信息增益

- 整合

**分类特征**

- 独热-one hot

- 连续值的特征

**推广**

- 回归树-regression trees

- 集成树-tree ensembles

- 有放回抽样-sampling with replacement

- 随机森林法

- XGBoost-eXtreme Gradient Boosting

**什么时候用决策树**

[第八章-聚类算法与异常检测](./第八章-聚类算法与异常检测.md)

**前言**

**聚类-clustering**

- 聚类的介绍

**k-means均值聚类算法**

- k-means代码实现

- 代价函数-distortion function

- k-means 初始化

- 集群数量的选择

**异常检测-Anomaly detection**

- 异常检测介绍

- 高斯正态分布-Gaussian distribution

- 异常检测算法

- 异常检测的设计与评估

**选择技巧**

- 监督学习vs异常检测

- 特征的选择

[第九章-推荐系统(协同与内容过滤)](./第九章-推荐系统(协同与内容过滤).md)

**前言**

**基于协同的过滤算法-collaborative filtering algorthm**

- 推荐系统的代价函数

- 协同过滤算法-Collaborative Filtering algorithm

- 二值标签-binary labels

**推荐系统的实现**

- 行的归一化

- tf实现协同过滤

- 找到相似商品

**基于内容的过滤算法-content-based filtering algorthm**

- 内容过滤vs协同过滤

- 深度学习实现内容过滤

- 从大目录中推荐

- tf中实现内容过滤

**社会伦理问题**

[第十章-强化学习与算法改进](./第十章-强化学习与算法改进.md)

**前言-什么是强化学习**

**强化学习**

- 案例：Mars rover example

- 强化学习中的回报-Return

- 强化学习中的策略

- 复习关键概念-马尔可夫决策过程MDP

**模型中的递归**

- 状态值函数-State action value function

- 案例：State-action value function example

- 贝尔曼方程-Bellman Equation

- 随机马尔可夫过程

**连续状态空间**

- 应用案例

- 案例-登陆月球

- 案例介绍

- 状态值函数

- 改进算法：神经网络架构

- 改进算法：$\epsilon$贪婪策略

- 改进算法：Mini-batch and soft update

**强化学习现状**

**完结撒花**

**附录-代码理解**
