# online-learning-optimizer
some online learning optimizer by pytorch

这些是之前做浙江省院高速公路流量预测实现的几个在线学习优化器。
* Online gradient descent(OGD) 说起来高级，其实就是在在线学习中直接使用SGD算法对模型参数进行更新，torch自带的库中都有，效果也不差
* follow the regularized leader(FTRL) 家族的在线学习算法很多，这里实现的应该是FTRL-Proximal。
* follow the moving leader (FTML) 在线学习器是基于github上一个大佬的代码改的（抱歉时间有点久了，不记得了），大佬的代码在一个地方错了，我进行了改正。
* sliding window gradient descent (SWGD) 是个有趣的算法，是一篇交通中的论文引入的，采用滑动时间窗口的形式，只有在这个窗口中所有梯度方向相同才进行更新，是一种比较保守的方法。其中初始化参数中：lr是学习率，N是时间窗口的大小

所有代码的使用可以直接import
