---
layout: post
title:  "Understanding Normal Equation"
subtitle: "什么是 Normal Equation? 如何推导? 有什么深层含义?"
date:   2019-01-10 23:45:00 +0800
background: '/img/posts/2019-01-10.jpg'
---

# 理解 Normal Equation

Normal Equation 是线性回归方程中,一种求最佳拟合系数的最小二乘方法.与机器学习中常见的多次迭代,逐渐逼近的方式不同,Normal Equation可以直接通过矩阵运算求得拟合系数.

$$\theta=(X^TX)^{-1}X^Ty\tag{1}$$
其中 $\theta$ 为待估参数:
$$
\theta=
\begin{bmatrix}
    \theta_0 & \theta_1 & \theta_2 & \cdots & \theta_n \\
\end{bmatrix}^T\tag{2}
$$

$X$,$y$ 为写成矩阵形式的数据,

$$
X=
\begin{bmatrix}
    x^{(1)} \\ x^{(2)} \\ \vdots \\ x^{(m)} \\
\end{bmatrix}=
\begin{bmatrix}
    x^{(1)}_1 & x^{(1)}_2 & \cdots & x^{(1)}_n \\
    x^{(2)}_1 & x^{(2)}_2 & \cdots & x^{(2)}_n \\
    \vdots    & \vdots    & \ddots & \vdots    \\
    x^{(m)}_1 & x^{(m)}_2 & \cdots & x^{(m)}_n \\
\end{bmatrix}
\quad \quad
y=
\begin{bmatrix}
    y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \\
\end{bmatrix}\tag{3}
$$

<font size = '1'>

注：为避免混淆,本文中所有涉及到的矩阵计算均采用分子布局,详情请参考
[Matrix Calculus][1].

</font>

<br/>

## 1. 推导过程

下面依次介绍3种推导方法

### 1.1 硬算

最简单粗暴的方法.</br>
根据最小二乘的定义,求 $\theta$ 使得预测值与实际观测值的均方误差最小,也即

$$\overline{\theta} = arg\ min(\frac{1}{2m}\sum_{i=1}^m (x^{(i)} \cdot \theta\ -y_i)^2)\tag{(4)}$$

右边括号中的值即为均方误差

$$\frac{1}{2m}\sum_{i=1}^m (x^{(i)} \cdot \theta\ -y_i)^2)$$

<font size = '1'>

注：此处为了后续求导时方便计算,将系数的分母乘以了2

</font>

上式是一个关于 $\theta$ 的二次多项式,明显存在最小值,且在 $\theta$ 的导数为0处取得.对 $\theta$ 的各个分量求导,可得

$$
\begin{cases}
\frac{1}{m}\sum_{i=1}^m (x^{(i)} \cdot \theta\ -y^{(i)})x_0^{(i)}=0 \\
\frac{1}{m}\sum_{i=1}^m (x^{(i)} \cdot \theta\ -y^{(i)})x_1^{(i)}=0 \\
\quad \cdots \\
\frac{1}{m}\sum_{i=1}^m (x^{(i)} \cdot \theta\ -y^{(i)})x_n^{(i)}=0 \\
\end{cases}
$$

写为矩阵形式即为

$$
\frac{1}{m}
\begin{bmatrix}
    x^{(1)}\cdot\theta-y^{(1)} \\
    x^{(2)}\cdot\theta-y^{(2)} \\
    \cdots \\
    x^{(m)}\cdot\theta-y^{(m)} \\
\end{bmatrix}^T
\cdot
\begin{bmatrix}
    x_0^{(1)} & x_1^{(1)} & \cdots & x_n^{(1)} \\
    x_0^{(2)} & x_1^{(2)} & \cdots & x_n^{(2)} \\
    \vdots    & \vdots    & \ddots & \vdots    \\
    x_0^{(m)} & x_1^{(m)} & \cdots & x_n^{(m)} \\
\end{bmatrix}=
\begin{bmatrix}
    0 & 0 & \cdots & 0 \\
\end{bmatrix}_{m \times 1}
$$

约去 $\frac{1}{m}$ ,将 $(2),(3)$ 式代入,并两边取逆,可得

$$
X^T(X\theta-y)=0
$$
$$
X^TX\theta=X^Ty
$$
$$
\theta=(X^TX)^{-1}X^Ty
$$

也就得到了 Normal Equation $(1)$.

<font size = '1'>

*后续会讨论 $X^TX$ 不可逆的情况.

</font>

</br>

### 1.2 矩阵微积分

同之前的思路相同,对 $\theta$ 求导,在其导数为0处取得最小值.但是不需要展开矩阵,可以直接通过矩阵微积分的方法,将 $\theta$ 看作一个整体进行求导.

<font size = '1'>

注：矩阵微积分是一种对多变量微积分的标记表示方法,与前述的硬算法其实没有本质上的区别,只是在引进了矩阵微积分的符号系统后,简化了计算推导过程.

</font>

将 $(4)$ 写成矩阵形式即为

$$\frac{1}{2m}(X\theta-y)^T \cdot (X\theta-y))\tag{5}$$

忽略 $\frac{1}{2m}$ ,展开为

$$
\theta^TX^TX\theta - \theta^TX^Ty-y^TX\theta+y^2
$$

<font size = '1'>

注：此式中各项皆为实值标量,对向量求导也就是对向量的各分量求导,并按照该向量的shape返回结果.标量对向量求导具体可参考[Matrix Calculus][1],此处直接使用了相关的公式：

$$
{\frac{\partial{x^Tx}}{\partial{x}}} = x^T(A+A^T)
\quad ; \quad
{\frac{\partial{x^Ta}}{\partial{x}}} = a^T
\quad ; \quad
{\frac{\partial{ax}}{\partial{x}}} = a
$$

具体推导过程此处不再展开,可参考[matrix prop][2].</br>
千万注意,关于矩阵微积分,不同作者的矩阵排布方式可能不同,同样的表达式,分子布局和分母布局的结果可能会差个转置或运算顺序,务必确认具体采用的是哪种布局,不然极容易出错.
</font>

对 $\theta$ 求导,可得
$$
\theta^T(X^TX+X^TX)-y^TX-y^TX=0
$$
化简可得
$$
X^TX\theta=X^Ty
$$
$$
\theta=(X^TX)^{-1}X^Ty
$$
</br>

另外,也可以根据链式求导法则,直接对 $(8)$ 直接求导,得到

$$
\frac{\partial{((X\theta-y)^T \cdot (X\theta-y))}}{\partial{\theta}}=
\frac{\partial{((X\theta-y)^T \cdot (X\theta-y))}}{\partial{(X\theta-y)}} \cdot
\frac{\partial{(X\theta-y)}}{\partial{(\theta)}}=
$$
$$
2(X\theta-y)^T \cdot X=0
$$

化简后可得

$$
\theta=(X^TX)^{-1}X^Ty
$$
</br>

### 1.3 几何意义推导

在理想情况下,我们实际上是想找到一组 $\theta$ ,满足

$$X\theta=y\tag{6}$$
完美地刻画出X与y之间的线性关系,但是因为误差原因,这在实际中是基本无法得到的.所以退而求其次,我们希望最小化 $X\theta$ 与 $y$ 之间的差异.注意到 $X\theta$ 实际上是 $X$ 的列向量的线性组合,那么我们的目标也可以看作：在 $X$ 的列空间中找到一个向量,使其与y的$^*$距离最近.其实也就是寻找 $y$ 在 $X$ 的列空间中的投影.而若 $X\theta$ 为投影向量,那么误差向量 $(X\theta-y)$ 与 $X$ 的列向量必定是相互垂直的.于是有
$$
X^T(X\theta-y)=0
$$
$$
X^TX\theta=X^Ty\tag{7}
$$
$$
\theta=(X^TX)^{-1}X^Ty
$$

<font size = '1'>

$^*$习惯上,我们说的距离都是指欧氏距离,这其实也是最小二乘中平方的本质.

</font>
</br>
</br>

## 2. 更进一步

回到 $(6)$ ,我们最朴素但也最容易落空的想法是根据此式直接解出 $\theta$ ,可是普遍的情况下, $m>n$ ,这是个超定方程组, $X_{m \times n}$ 的逆不存在.

但是再看 $(7)$ ,这是得出Normal Equation 的前一步,然而从形式上看,只是在 $(6)$ 的等式两端分别左乘了 $X^T$ .为什么这样就能求得超定方程组 $(9)$ 的最小二乘解？

回到 $X\theta=y$ ,若 $X$ 可逆,那么可以直接等式两边左乘 $X^{-1}$ ,得到 $\theta=X^{-1}y$, 而当 $X$ 不可逆的时候,想要达到类似的效果,可以采用 $X$ 的伪逆 $X^+$ (或称作广义逆),其中最广为人知的即为 Moore–Penrose inverse,具体可以参考wiki [[Moore–Penrose inverse]][4].

根据其定义, $X^+$ 需满足下述条件中的1个或多个:

$$
XX^+X=X\tag{a}
$$

$$
X^+XX^+=X^+\tag{b}
$$
$$
(XX^+)^*=XX^+\tag{c}
$$
$$
(X^+X)^*=X^+X\tag{d}
$$

<font size = '1'>

注：由于我们讨论的 $X$ 都是实值矩阵,所以 $A^*=A^T$ .

</font>

将 $(c)$ 展开,得到
$$(X^+)^TX^T=XX^+$$
两端右乘 $X$
$$(X^+)^TX^TX=XX^+X$$
将 $(1)$ 代入右端
$$(X^+)^TX^TX=X$$
$$X^TXX^+=X^T$$
注意到当 $X$ 的列向量线性独立时,
$X_{n \times m}^T \cdot X_{n \times m}$
是一个 $n \times n$ 且秩为n的方阵,可逆,于是有
$$X^+=(X^TX)^{-1}X^T$$

$$
X^+=(X^TX)^{-1}X^T\tag{8}
$$
根据前述思路有

$$
\theta=X^+y=(X^TX)^{-1}X^Ty
$$

注意到我们其实只用到了4条性质中的 $(1)(3)$ 这2条性质,因为其实我们要求的是 $X$ 的左逆,也即找到一个矩阵,使得 $X$ 在其表示的行变换下,变为单位矩阵.

而 $(a)$ 可以看作 $(XX^+)X=(I)X$ ,意义恰恰是在$X$行空间中,上述命题成立.我们只需在此基础上对 $XX^+$ 做处理即可,而 $(3)$ 正是关于 $XX^+$ 的等式.

<font size = '1'>

思考一下,广义右逆怎么推导？

</font>
</br>

总之,我们又一次得到了 Normal Equation.也就是说,当在 $(9)$ 的两边分别左乘 $X^T$ 的时候,恰好满足了 $X$ 的伪逆的形式.那么左乘其他矩阵,是否能达到类似的效果呢？答案是肯定的.

</br>
</br>

## 3. 再进一步

注意到 $(8)$ 的成立要求 $X_{n \times m}^T \cdot X_{n \times m}$ 是可逆的,而我们的讨论范围限定在 $m>n$ 的条件下,并且我们知道 $X^TX$与$X$ 是相抵的,即

$$rank(X^TX)=rank(X)=rank(X^T)=n$$

所以在 $X$ 列满秩的情况下, $(X^TX)_{n \times n}$ 确实是可逆的.

也就是说,在 $X\theta=y$ 也即 $X\theta=Iy$ 的两侧分别左乘 $X$ 后,等式左侧并无信息的损失,因为秩没有降.只在等式右侧可能有信息损失.

<font size = '1'>

直觉是等式右侧在左乘 $X$ 时相当于做了一次投影,投影时损失掉的那部分error_vector,但是没有仔细研究,以后有时间了补上~

</font>

所以关键是让等式左侧在左乘一个矩阵之后,变得可逆,这就要求左乘的矩阵的shape应该与 $X^T$ 相同(从而保证运算后是个方阵),同时保持其秩不变.这样的矩阵除了 $X^T$ 外,还存在吗？当然存在.

考虑 $X$ 的QR分解
$$
X_{m \times n} = Q_{m \times n} \cdot R_{n \times n}
$$
其中 $Q_{m \times n}$ 列满秩正交矩阵, $R_{n \times n}$ 为n阶满秩方阵. $Q^T$ 即为我们要找的矩阵.

$$
X\theta=y
$$

$$
QR\theta=y
$$

$$
Q^TQR\theta=Q^Ty
$$

$$
IR\theta=Q^Ty
$$

$$
\theta=R^{-1}Q^Ty
$$

注意左乘矩阵的选择并不惟一,比如 $X^T$ ,又比如 $Q$ 经初等变换得到的矩阵均满足要求.但QR分解计算效率高, $Q^TQ=I$ 这个性质大大简化了计算,并且有很多成熟的算法和包可以高效实现这一过程,所以这也是目前在解线性回归模型时最常用的方法之一.
</br>
</br>

## 4. 小结

综上,用最小二乘法处理线性回归问题实际上就是寻找 $y$ 在某一线性空间中的投影向量,这个空间最直观的选择就是 $X$ 本身的列空间,而为了提高计算速度,常选用 $X$ 经QR分解后的得到的正交矩阵 $Q$ 的列空间.

</br>
</br>

## Reference

<font size = '2'>

> Matrix Calculus:
></br>
>[https://en.wikipedia.org/wiki/Matrix_calculus/](ttps://en.wikipedia.org/wiki/Matrix_calculus/)

</br>

> Properties of the Trace and Matrix Derivatives:
></br>
> [https://web.stanford.edu/~jduchi/projects/matrix_prop.pdf/](https://web.stanford.edu/~jduchi/projects/matrix_prop.pdf/)

</br>

> Moore–Penrose inverse:
></br>
>[https://en.wikipedia.org/wiki/Moore–Penrose_inverse/](https://en.wikipedia.org/wiki/Moore–Penrose_inverse)

</br>

> 掰开揉碎推导Normal Equation:
></br>  
> [https://zhuanlan.zhihu.com/p/22757336/](https://zhuanlan.zhihu.com/p/22757336)

</br>

> 5种方法推导 Normal Equation:
></br>
>[https://www.cnblogs.com/AngelaSunny/p/6616712.html/](https://www.cnblogs.com/AngelaSunny/p/6616712.html)

</font>

[1]:https://en.wikipedia.org/wiki/Matrix_calculus/ "Matrix Calculus"
[2]:https://web.stanford.edu/~jduchi/projects/matrix_prop.pdf/ "Properties of the Trace and Matrix Derivatives"
[3]:https://zhuanlan.zhihu.com/p/22757336/ "zhihu"
[4]:https://en.wikipedia.org/wiki/Moore–Penrose_inverse "Moore–Penrose inverse"
[5]:https://www.cnblogs.com/AngelaSunny/p/6616712.html "5种方法推导 Normal Equation"