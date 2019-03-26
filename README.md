# house_price
房价预测

一、过程分析

1. 读取相关数据
2. 数据分析及处理
（1）通过对业务的理解，选取比较重要的元素进行分析处理
（2）分析及补全缺失数据
（3）其他针对具体元素的处理，如类别变量数字化LabelEncoder、向量化get_dummies
（4）一些特征重组或拆分
3. 数据建模
（1）处理变量间的线性关系Lasso
（2）单个预测模型效果不好，考虑堆叠模型
（3）ElasticNet、KernelRidge、GradientBoostingRegressor、XGBRegressor

二、画图
1. 散点图matplotlib.pyplot scatter：展示面积与房价的关系
2. 直方图seaborn.distplot:房价分布
3. 箱型图seaborn.boxplot:房屋质量等级对房价的影响；建造年份对房价影响
4. 热图seaborn.heatmap:各元素间的相关性
