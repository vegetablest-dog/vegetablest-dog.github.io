---
title: Note for AFML C.6 - 集成方法
tags:
  - modeling
---
## 集成方法
Bagging可以降低方差，但不能提高准确率。
如果每个时间点 t 的观察值按 t 和 t+100 之间的收益来标记，我们应该为每个装袋估计器抽取 1% 的观察数据，但不要更多。
### 随机森林
防止过拟合的方法：
1. 将参数 max_features 设置为较低的值，以此作为强制树之间出现差异的一种方法
2. 早停：将正则化参数 min_weight_fraction_leaf 设置为足够大的值（例如 5%），以使袋外准确率收敛到样本外（k 折）准确率。
3. 在 DecisionTreeClassifier 上使用 BaggingClassifier，其中 max_samples 设置为样本之间的平均唯一性 (avgU)。
	```
	clf=DecisionTreeClassifier(criterion='entropy',max_ features='auto',class_weight='balanced')
	bc=BaggingClassifier(base_estimator=clf,n_estimators= 1000,max_samples=avgU,max_features=1.)
	```
4. 在 RandomForestClassifier 上使用 BaggingClassifier，其中 max_samples 设置为样本之间的平均唯一性（avgU）。
	```
	clf=RandomForestClassifier(n_estimators=1,criterion= 'entropy',bootstrap=False,class_weight='balanced_ subsample')
	bc=BaggingClassifier(base_estimator=clf,n_estimators= 1000,max_samples=avgU,max_features=1.)
	```
5. 修改 RF 类，将标准自助法替换为顺序自助法。
在拟合决策树时，将特征空间旋转到与坐标轴对齐的方向通常可以减少树所需的层数。因此，我建议你在对特征进行 PCA 后再拟合随机森林，因为这可能加快计算速度并减少一些过拟合
### 增强方法在金融中
bagging和boosting主要有以下几点不同：
- 各个分类器是依次进行拟合的。
- 表现不佳的分类器会被淘汰。
- 每次迭代中，观察值的权重不同。
- 集成预测是各个学习器的加权平均。
提升的主要优势在于它可以减少预测中的方差和偏差。然而，纠正偏差的代价是增加了过拟合的风险。
可以认为，在金融应用中，通常袋装法比提升法更可取。袋装法解决的是过拟合问题，而提升法解决的是欠拟合问题。过拟合通常比欠拟合更值得关注，因为由于信噪比低，机器学习算法很容易在金融数据上发生过拟合。此外，袋装法可以并行处理，而通常提升法需要顺序运行。
### 为了可扩展性进行装袋
支持向量机（SVM）就是一个典型例子。如果你尝试在一百万个样本上拟合SVM，算法可能需要一段时间才能收敛。即便算法收敛了，也不能保证所得到的解是全局最优，或者没有过拟合。
一种实际的方法是构建一个装袋算法，其中基础估计器属于对样本量不易扩展的类别。
鉴于装袋算法可以并行化，我们将一个大型的顺序任务转换为许多可以同时运行的小任务。当然，提前停止会增加各个基础估计器输出的方差；然而，这种增加可以被装袋算法带来的方差降低所弥补。你可以通过增加更多独立的基础估计器来控制这种降低。以这种方式使用时，装袋算法将使你能够在极其大型的数据集上获得快速且稳健的估计。


