##第一周  简单介绍，单变量线性回归，梯度下降，线性代数回顾



##资源
- wiki https://share.coursera.org/wiki/index.php/ML:Main#Week_1



##最小二乘法
http://www.cnblogs.com/iamccme/archive/2013/05/15/3080737.html



##最小二乘法和梯度下降法有哪些区别
最小二乘法的目标：求误差的最小平方和，对应有两种：线性和非线性。线性最小二乘的解是closed-form即，而非线性最小二乘没有closed-form，通常用迭代法求解。

迭代法，即在每一步update未知量逐渐逼近解，可以用于各种各样的问题（包括最小二乘），比如求的不是误差的最小平方和而是最小立方和。

梯度下降是迭代法的一种，可以用于求解最小二乘问题（线性和非线性都可以）。高斯-牛顿法是另一种经常用于求解非线性最小二乘的迭代法（一定程度上可视为标准非线性最小二乘求解方法）。

还有一种叫做Levenberg-Marquardt的迭代法用于求解非线性最小二乘问题，就结合了梯度下降和高斯-牛顿法。

所以如果把最小二乘看做是优化问题的话，那么梯度下降是求解方法的一种，是求解线性最小二乘的一种，高斯-牛顿法和Levenberg-Marquardt则能用于求解非线性最小二乘。

具体可参考维基百科（Least squares, Gradient descent, Gauss-Newton algorithm, Levenberg-Marquardt algorithm）

作者：夏之晨
链接：https://www.zhihu.com/question/20822481/answer/23648885
来源：知乎


