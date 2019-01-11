---
layout: post
title: 'Understanding Normal Equation'
subtitle: '什么是 Normal Equation? 如何推导? 有什么深层含义?'
date: '2019-01-11 23:45:00 +0800'
background: /img/posts/bridge_lines.jpg
---  
  
  
# 理解 Normal Equation
  
  
Normal Equation 是线性回归方程中,一种求最佳拟合系数的最小二乘方法.与机器学习中常见的多次迭代,逐渐逼近的方式不同,Normal Equation可以直接通过矩阵运算求得拟合系数.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta=(X^TX)^{-1}X^Ty&#x5C;tag{1}"/></p>  
  
其中 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> 为待估参数:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta=&#x5C;begin{bmatrix}%20%20%20%20&#x5C;theta_0%20&amp;%20&#x5C;theta_1%20&amp;%20&#x5C;theta_2%20&amp;%20&#x5C;cdots%20&amp;%20&#x5C;theta_n%20&#x5C;&#x5C;&#x5C;end{bmatrix}^T&#x5C;tag{2}"/></p>  
  
  
<img src="https://latex.codecogs.com/gif.latex?X"/>,<img src="https://latex.codecogs.com/gif.latex?y"/> 为写成矩阵形式的数据,
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X=&#x5C;begin{bmatrix}%20%20%20%20x^{(1)}%20&#x5C;&#x5C;%20x^{(2)}%20&#x5C;&#x5C;%20&#x5C;vdots%20&#x5C;&#x5C;%20x^{(m)}%20&#x5C;&#x5C;&#x5C;end{bmatrix}=&#x5C;begin{bmatrix}%20%20%20%20x^{(1)}_1%20&amp;%20x^{(1)}_2%20&amp;%20&#x5C;cdots%20&amp;%20x^{(1)}_n%20&#x5C;&#x5C;%20%20%20%20x^{(2)}_1%20&amp;%20x^{(2)}_2%20&amp;%20&#x5C;cdots%20&amp;%20x^{(2)}_n%20&#x5C;&#x5C;%20%20%20%20&#x5C;vdots%20%20%20%20&amp;%20&#x5C;vdots%20%20%20%20&amp;%20&#x5C;ddots%20&amp;%20&#x5C;vdots%20%20%20%20&#x5C;&#x5C;%20%20%20%20x^{(m)}_1%20&amp;%20x^{(m)}_2%20&amp;%20&#x5C;cdots%20&amp;%20x^{(m)}_n%20&#x5C;&#x5C;&#x5C;end{bmatrix}&#x5C;quad%20&#x5C;quady=&#x5C;begin{bmatrix}%20%20%20%20y^{(1)}%20&#x5C;&#x5C;%20y^{(2)}%20&#x5C;&#x5C;%20&#x5C;vdots%20&#x5C;&#x5C;%20y^{(m)}%20&#x5C;&#x5C;&#x5C;end{bmatrix}&#x5C;tag{3}"/></p>  
  
  
<font size = '1'>
  
注：为避免混淆,本文中所有涉及到的矩阵计算均采用分子布局,详情请参考 [Matrix Calculus][1].
  
</font>
  
<br/>
  
## 1. 推导过程
  
  
下面依次介绍3种推导方法
  
### 1.1 硬算
  
  
最简单粗暴的方法.<br/>
根据最小二乘的定义,求 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> 使得预测值与实际观测值的均方误差最小,也即
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;overline{&#x5C;theta}%20=%20arg&#x5C;%20min(&#x5C;frac{1}{2m}&#x5C;sum_{i=1}^m%20(x^{(i)}%20&#x5C;cdot%20&#x5C;theta&#x5C;%20-y_i)^2)&#x5C;tag{4}"/></p>  
  
  
右边括号中的值即为均方误差
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{1}{2m}&#x5C;sum_{i=1}^m%20(x^{(i)}%20&#x5C;cdot%20&#x5C;theta&#x5C;%20-y_i)^2)"/></p>  
  
  
<font size = '1'>
  
注：此处为了后续求导时方便计算,将系数的分母乘以了2
  
</font>
  
上式是一个关于 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> 的二次多项式,明显存在最小值,且在 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> 的导数为0处取得.对 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> 的各个分量求导,可得
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{cases}&#x5C;frac{1}{m}&#x5C;sum_{i=1}^m%20(x^{(i)}%20&#x5C;cdot%20&#x5C;theta&#x5C;%20-y^{(i)})x_0^{(i)}=0%20&#x5C;&#x5C;&#x5C;frac{1}{m}&#x5C;sum_{i=1}^m%20(x^{(i)}%20&#x5C;cdot%20&#x5C;theta&#x5C;%20-y^{(i)})x_1^{(i)}=0%20&#x5C;&#x5C;&#x5C;quad%20&#x5C;cdots%20&#x5C;&#x5C;&#x5C;frac{1}{m}&#x5C;sum_{i=1}^m%20(x^{(i)}%20&#x5C;cdot%20&#x5C;theta&#x5C;%20-y^{(i)})x_n^{(i)}=0%20&#x5C;&#x5C;&#x5C;end{cases}"/></p>  
  
  
写为矩阵形式即为

<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{1}{m}&#x5C;begin{bmatrix}%20%20%20%20x^{(1)}&#x5C;cdot&#x5C;theta-y^{(1)}%20&#x5C;&#x5C;%20%20%20%20x^{(2)}&#x5C;cdot&#x5C;theta-y^{(2)}%20&#x5C;&#x5C;%20%20%20%20&#x5C;cdots%20&#x5C;&#x5C;%20%20%20%20x^{(m)}&#x5C;cdot&#x5C;theta-y^{(m)}%20&#x5C;&#x5C;&#x5C;end{bmatrix}^T&#x5C;cdot&#x5C;begin{bmatrix}%20%20%20%20x_0^{(1)}%20&amp;%20x_1^{(1)}%20&amp;%20&#x5C;cdots%20&amp;%20x_n^{(1)}%20&#x5C;&#x5C;%20%20%20%20x_0^{(2)}%20&amp;%20x_1^{(2)}%20&amp;%20&#x5C;cdots%20&amp;%20x_n^{(2)}%20&#x5C;&#x5C;%20%20%20%20&#x5C;vdots%20%20%20%20&amp;%20&#x5C;vdots%20%20%20%20&amp;%20&#x5C;ddots%20&amp;%20&#x5C;vdots%20%20%20%20&#x5C;&#x5C;%20%20%20%20x_0^{(m)}%20&amp;%20x_1^{(m)}%20&amp;%20&#x5C;cdots%20&amp;%20x_n^{(m)}%20&#x5C;&#x5C;&#x5C;end{bmatrix}=&#x5C;begin{bmatrix}%20%20%20%200%20&amp;%200%20&amp;%20&#x5C;cdots%20&amp;%200%20&#x5C;&#x5C;&#x5C;end{bmatrix}_{m%20&#x5C;times%201}"/></p>  
  
  
约去 <img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{1}{m}"/> ,将 <img src="https://latex.codecogs.com/gif.latex?(2),(3)"/> 式代入,并两边取逆,可得
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X^T(X&#x5C;theta-y)=0"/></p>  
  
可以解得
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta=(X^TX)^{-1}X^Ty"/></p>  
  
  
也就得到了 Normal Equation <img src="https://latex.codecogs.com/gif.latex?(1)"/>.
  
<font size = '1'>
  
*后续会讨论 <img src="https://latex.codecogs.com/gif.latex?X^TX"/> 不可逆的情况.
  
</font>
  
<br/>
  
### 1.2 矩阵微积分
  
  
同之前的思路相同,对 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> 求导,在其导数为0处取得最小值.但是不需要展开矩阵,可以直接通过矩阵微积分的方法,将 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> 看作一个整体进行求导.
  
<font size = '1'>
  
注：矩阵微积分是一种对多变量微积分的标记表示方法,与前述的硬算法其实没有本质上的区别,只是在引进了矩阵微积分的符号系统后,简化了计算推导过程.
  
</font>
  
将 <img src="https://latex.codecogs.com/gif.latex?(4)"/> 写成矩阵形式即为
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{1}{2m}(X&#x5C;theta-y)^T%20&#x5C;cdot%20(X&#x5C;theta-y))&#x5C;tag{5}"/></p>  
  
  
忽略 <img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{1}{2m}"/> ,展开为
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta^TX^TX&#x5C;theta%20-%20&#x5C;theta^TX^Ty-y^TX&#x5C;theta+y^2"/></p>  
  
  
<font size = '1'>
  
注：此式中各项皆为实值标量,对向量求导也就是对向量的各分量求导,并按照该向量的shape返回结果.标量对向量求导具体可参考 [Matrix Calculus][1] ,此处直接使用了相关的公式：
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?{&#x5C;frac{&#x5C;partial{x^Tx}}{&#x5C;partial{x}}}%20=%20x^T(A+A^T)&#x5C;quad%20;%20&#x5C;quad{&#x5C;frac{&#x5C;partial{x^Ta}}{&#x5C;partial{x}}}%20=%20a^T&#x5C;quad%20;%20&#x5C;quad{&#x5C;frac{&#x5C;partial{ax}}{&#x5C;partial{x}}}%20=%20a"/></p>  
  
  
具体推导过程此处不再展开,可参考 [Properties of the Trace and Matrix Derivatives][2].<br/>
千万注意,关于矩阵微积分,不同作者的矩阵排布方式可能不同,同样的表达式,分子布局和分母布局的结果可能会差个转置或运算顺序,务必确认具体采用的是哪种布局,不然极容易出错.
</font>
  
对 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> 求导,可得
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta^T(X^TX+X^TX)-y^TX-y^TX=0"/></p>  
  
  
化简可得
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X^TX&#x5C;theta=X^Ty"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta=(X^TX)^{-1}X^Ty"/></p>  
  
<br/>
  
另外,也可以根据链式求导法则,直接对 <img src="https://latex.codecogs.com/gif.latex?(8)"/> 直接求导,得到
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial{((X&#x5C;theta-y)^T%20&#x5C;cdot%20(X&#x5C;theta-y))}}{&#x5C;partial{&#x5C;theta}}=&#x5C;frac{&#x5C;partial{((X&#x5C;theta-y)^T%20&#x5C;cdot%20(X&#x5C;theta-y))}}{&#x5C;partial{(X&#x5C;theta-y)}}%20&#x5C;cdot&#x5C;frac{&#x5C;partial{(X&#x5C;theta-y)}}{&#x5C;partial{(&#x5C;theta)}}=2(X&#x5C;theta-y)^T%20&#x5C;cdot%20X=0"/></p>  
  
  
化简后可得
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta=(X^TX)^{-1}X^Ty"/></p>  
  
<br/>
  
### 1.3 几何意义推导
  
  
在理想情况下,我们实际上是想找到一组 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> ,满足
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X&#x5C;theta=y&#x5C;tag{6}"/></p>  
  
完美地刻画出X与y之间的线性关系,但是因为误差原因,这在实际中是基本无法得到的.所以退而求其次,我们希望最小化 <img src="https://latex.codecogs.com/gif.latex?X&#x5C;theta"/> 与 <img src="https://latex.codecogs.com/gif.latex?y"/> 之间的差异.注意到 <img src="https://latex.codecogs.com/gif.latex?X&#x5C;theta"/> 实际上是 <img src="https://latex.codecogs.com/gif.latex?X"/> 的列向量的线性组合,那么我们的目标也可以看作：在 <img src="https://latex.codecogs.com/gif.latex?X"/> 的列空间中找到一个向量,使其与y的<img src="https://latex.codecogs.com/gif.latex?^*"/>距离最近.其实也就是寻找 <img src="https://latex.codecogs.com/gif.latex?y"/> 在 <img src="https://latex.codecogs.com/gif.latex?X"/> 的列空间中的投影.而若 <img src="https://latex.codecogs.com/gif.latex?X&#x5C;theta"/> 为投影向量,那么误差向量 <img src="https://latex.codecogs.com/gif.latex?(X&#x5C;theta-y)"/> 与 <img src="https://latex.codecogs.com/gif.latex?X"/> 的列向量必定是相互垂直的.于是有
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X^T(X&#x5C;theta-y)=0"/></p>  
  
  
可以解得
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta=(X^TX)^{-1}X^Ty"/></p>  
  
  
<font size = '1'>
  
<img src="https://latex.codecogs.com/gif.latex?^*"/>习惯上,我们说的距离都是指欧氏距离,这其实也是最小二乘中平方的本质.
  
</font>
<br/>
<br/>
  
## 2. 更进一步
  
  
回到 <img src="https://latex.codecogs.com/gif.latex?(6)"/> ,我们最朴素但也最容易落空的想法是根据此式直接解出 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> ,可是普遍的情况下, <img src="https://latex.codecogs.com/gif.latex?m&gt;n"/> ,这是个超定方程组, <img src="https://latex.codecogs.com/gif.latex?X_{m%20&#x5C;times%20n}"/> 的逆不存在.
  
但是再看 <img src="https://latex.codecogs.com/gif.latex?(7)"/> ,这是得出Normal Equation 的前一步,然而从形式上看,只是在 <img src="https://latex.codecogs.com/gif.latex?(6)"/> 的等式两端分别左乘了 <img src="https://latex.codecogs.com/gif.latex?X^T"/> .为什么这样就能求得超定方程组 <img src="https://latex.codecogs.com/gif.latex?(9)"/> 的最小二乘解？
  
回到 <img src="https://latex.codecogs.com/gif.latex?X&#x5C;theta=y"/> ,若 <img src="https://latex.codecogs.com/gif.latex?X"/> 可逆,那么可以直接等式两边左乘 <img src="https://latex.codecogs.com/gif.latex?X^{-1}"/> ,得到 <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta=X^{-1}y"/>, 而当 <img src="https://latex.codecogs.com/gif.latex?X"/> 不可逆的时候,想要达到类似的效果,可以采用 <img src="https://latex.codecogs.com/gif.latex?X"/> 的伪逆 <img src="https://latex.codecogs.com/gif.latex?X^+"/> (或称作广义逆),其中最广为人知的即为 Moore–Penrose inverse,具体可以参考 [Moore–Penrose inverse][3] .
  
根据其定义, <img src="https://latex.codecogs.com/gif.latex?X^+"/> 需满足下述条件中的1个或多个:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?XX^+X=X&#x5C;tag{a}"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X^+XX^+=X^+&#x5C;tag{b}"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?(XX^+)^*=XX^+&#x5C;tag{c}"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?(X^+X)^*=X^+X&#x5C;tag{d}"/></p>  
  
  
<font size = '1'>
  
注：由于我们讨论的 <img src="https://latex.codecogs.com/gif.latex?X"/> 都是实值矩阵,所以 <img src="https://latex.codecogs.com/gif.latex?A^*=A^T"/> .
  
</font>
  
将 <img src="https://latex.codecogs.com/gif.latex?(c)"/> 展开,得到
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?(X^+)^TX^T=XX^+"/></p>  
  
  
两端右乘 <img src="https://latex.codecogs.com/gif.latex?X"/>
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?(X^+)^TX^TX=XX^+X"/></p>  
  
  
将 <img src="https://latex.codecogs.com/gif.latex?(1)"/> 代入右端
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?(X^+)^TX^TX=X"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X^TXX^+=X^T"/></p>  
  
  
注意到当 <img src="https://latex.codecogs.com/gif.latex?X"/> 的列向量线性独立时,
<img src="https://latex.codecogs.com/gif.latex?X_{n%20&#x5C;times%20m}^T%20&#x5C;cdot%20X_{n%20&#x5C;times%20m}"/>
是一个 <img src="https://latex.codecogs.com/gif.latex?n%20&#x5C;times%20n"/> 且秩为n的方阵,可逆,于是有
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X^+=(X^TX)^{-1}X^T"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X^+=(X^TX)^{-1}X^T&#x5C;tag{8}"/></p>  
  
  
根据前述思路有
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta=X^+y=(X^TX)^{-1}X^Ty"/></p>  
  
  
注意到我们其实只用到了4条性质中的 <img src="https://latex.codecogs.com/gif.latex?(1)(3)"/> 这2条性质,因为其实我们要求的是 <img src="https://latex.codecogs.com/gif.latex?X"/> 的左逆,也即找到一个矩阵,使得 <img src="https://latex.codecogs.com/gif.latex?X"/> 在其表示的行变换下,变为单位矩阵.
  
而 <img src="https://latex.codecogs.com/gif.latex?(a)"/> 可以看作 <img src="https://latex.codecogs.com/gif.latex?(XX^+)X=(I)X"/> ,意义恰恰是在<img src="https://latex.codecogs.com/gif.latex?X"/>行空间中,上述命题成立.我们只需在此基础上对 <img src="https://latex.codecogs.com/gif.latex?XX^+"/> 做处理即可,而 <img src="https://latex.codecogs.com/gif.latex?(3)"/> 正是关于 <img src="https://latex.codecogs.com/gif.latex?XX^+"/> 的等式.
  
<font size = '1'>
  
思考一下,广义右逆怎么推导？
  
</font>
<br/>
  
总之,我们又一次得到了 Normal Equation.也就是说,当在 <img src="https://latex.codecogs.com/gif.latex?(9)"/> 的两边分别左乘 <img src="https://latex.codecogs.com/gif.latex?X^T"/> 的时候,恰好满足了 <img src="https://latex.codecogs.com/gif.latex?X"/> 的伪逆的形式.那么左乘其他矩阵,是否能达到类似的效果呢？答案是肯定的.
  
<br/>
<br/>
  
## 3. 再进一步
  
  
注意到 <img src="https://latex.codecogs.com/gif.latex?(8)"/> 的成立要求 <img src="https://latex.codecogs.com/gif.latex?X_{n%20&#x5C;times%20m}^T%20&#x5C;cdot%20X_{n%20&#x5C;times%20m}"/> 是可逆的,而我们的讨论范围限定在 <img src="https://latex.codecogs.com/gif.latex?m&gt;n"/> 的条件下,并且我们知道 <img src="https://latex.codecogs.com/gif.latex?X^TX"/>与<img src="https://latex.codecogs.com/gif.latex?X"/> 是相抵的,即
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?rank(X^TX)=rank(X)=rank(X^T)=n"/></p>  
  
  
所以在 <img src="https://latex.codecogs.com/gif.latex?X"/> 列满秩的情况下, <img src="https://latex.codecogs.com/gif.latex?(X^TX)_{n%20&#x5C;times%20n}"/> 确实是可逆的.
  
也就是说,在 <img src="https://latex.codecogs.com/gif.latex?X&#x5C;theta=y"/> 也即 <img src="https://latex.codecogs.com/gif.latex?X&#x5C;theta=Iy"/> 的两侧分别左乘 <img src="https://latex.codecogs.com/gif.latex?X"/> 后,等式左侧并无信息的损失,因为秩没有降.只在等式右侧可能有信息损失.
  
<font size = '1'>
  
直觉是等式右侧在左乘 <img src="https://latex.codecogs.com/gif.latex?X"/> 时相当于做了一次投影,投影时损失掉的那部分error_vector,但是没有仔细研究,以后有时间了补上~
  
</font>
  
所以关键是让等式左侧在左乘一个矩阵之后,变得可逆,这就要求左乘的矩阵的shape应该与 <img src="https://latex.codecogs.com/gif.latex?X^T"/> 相同(从而保证运算后是个方阵),同时保持其秩不变.这样的矩阵除了 <img src="https://latex.codecogs.com/gif.latex?X^T"/> 外,还存在吗？当然存在.
  
考虑 <img src="https://latex.codecogs.com/gif.latex?X"/> 的QR分解
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X_{m%20&#x5C;times%20n}%20=%20Q_{m%20&#x5C;times%20n}%20&#x5C;cdot%20R_{n%20&#x5C;times%20n}"/></p>  
  
  
其中 <img src="https://latex.codecogs.com/gif.latex?Q_{m%20&#x5C;times%20n}"/> 列满秩正交矩阵, <img src="https://latex.codecogs.com/gif.latex?R_{n%20&#x5C;times%20n}"/> 为n阶满秩方阵. <img src="https://latex.codecogs.com/gif.latex?Q^T"/> 即为我们要找的矩阵.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X&#x5C;theta=y"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?QR&#x5C;theta=y"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?Q^TQR&#x5C;theta=Q^Ty"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?IR&#x5C;theta=Q^Ty"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;theta=R^{-1}Q^Ty"/></p>  
  
  
注意左乘矩阵的选择并不惟一,比如 <img src="https://latex.codecogs.com/gif.latex?X^T"/> ,又比如 <img src="https://latex.codecogs.com/gif.latex?Q"/> 经初等变换得到的矩阵均满足要求.但QR分解计算效率高, <img src="https://latex.codecogs.com/gif.latex?Q^TQ=I"/> 这个性质大大简化了计算,并且有很多成熟的算法和包可以高效实现这一过程,所以这也是目前在解线性回归模型时最常用的方法之一.
<br/>
<br/>
  
## 4. 小结
  
  
综上,用最小二乘法处理线性回归问题实际上就是寻找 <img src="https://latex.codecogs.com/gif.latex?y"/> 在某一线性空间中的投影向量,这个空间最直观的选择就是 <img src="https://latex.codecogs.com/gif.latex?X"/> 本身的列空间,而为了提高计算速度,常选用 <img src="https://latex.codecogs.com/gif.latex?X"/> 经QR分解后的得到的正交矩阵 <img src="https://latex.codecogs.com/gif.latex?Q"/> 的列空间.
  
<br/>
<br/>
  
## Reference
  
  
<font size = '2'>
  
  
1. Matrix Calculus:
[https://en.wikipedia.org/wiki/Matrix_calculus/](https://en.wikipedia.org/wiki/Matrix_calculus/ )
<br/>
2. Properties of the Trace and Matrix Derivatives:
[https://web.stanford.edu/~jduchi/projects/matrix_prop.pdf/](https://web.stanford.edu/~jduchi/projects/matrix_prop.pdf/ )
<br/>
3. Moore–Penrose inverse:
[https://en.wikipedia.org/wiki/Moore–Penrose_inverse/](https://en.wikipedia.org/wiki/Moore–Penrose_inverse )
<br/>
4. 掰开揉碎推导Normal Equation:
[https://zhuanlan.zhihu.com/p/22757336/](https://zhuanlan.zhihu.com/p/22757336 )
<br/>
5. 5种方法推导 Normal Equation:
[https://www.cnblogs.com/AngelaSunny/p/6616712.html/](https://www.cnblogs.com/AngelaSunny/p/6616712.html )
<br/>
  
</font>
  
[1]:https://en.wikipedia.org/wiki/Matrix_calculus/ "Matrix Calculus"
[2]:https://web.stanford.edu/~jduchi/projects/matrix_prop.pdf/ "Properties of the Trace and Matrix Derivatives"
[3]:https://en.wikipedia.org/wiki/Moore–Penrose_inverse "Moore–Penrose inverse"
[4]:https://zhuanlan.zhihu.com/p/22757336/ "zhihu"
[5]:https://www.cnblogs.com/AngelaSunny/p/6616712.html "5种方法推导 Normal Equation"
  