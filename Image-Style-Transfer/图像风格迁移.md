# 图像风格迁移

![网络模型](https://images2018.cnblogs.com/blog/899363/201712/899363-20171225164109556-1235586668.png)

## 损失函数组成

> #### Loss = w1 * Lc + w2 * Ls

- ### **Loss of content(Lc)**

- ### **Loss of style(Ls)**

## Loss of content

> 内容图和随机噪声图经过多次卷积滤波后，conten和noise在第4层的feature map的距离的平方和

![Lc](https://img-blog.csdnimg.cn/20190220160543599.png)

## Loss of style

> 先对风格图和噪声图的每一层卷积得到feature map
>
> 对feature map求gram矩阵
>
> 计算两者gram距离的平方和
>
> 将5层的结果加权求和

![gram](https://img-blog.csdnimg.cn/20190220161017156.png)
![Ls](https://pic2.zhimg.com/80/v2-6ea00b4233e081855031bcb51899c7e9_1440w.jpg)

## 实验图

![卷积效果](https://images2018.cnblogs.com/blog/899363/201712/899363-20171225164620790-80364289.png)

> 随着卷积网络层数增加，获得的特征映射更加抽象。
>
> 上图可以看出，层数增高的时候：
>
> - 内容**重构图可变化性**增加，具有更大的风格变化能力。
>
> - 风格随着使用的层数越多，**风格迁移的稳定性越强**。

## Gram矩阵

### 定义

> n维欧式空间中任意k个向量之间两两的内积所组成的矩阵，称为这k个向量的**格拉姆矩阵*(Gram matrix)***，很明显，这是一个对称矩阵。

![gram](https://img2020.cnblogs.com/blog/1704791/202005/1704791-20200510091258297-1814861622.png)

![Gram](https://img2020.cnblogs.com/blog/1704791/202005/1704791-20200510091258621-1096842037.png)

### 计算

> 输入图像的feature map为**[ ch, h, w]**。
>
> 我们经过**flatten**和**矩阵转置**操作
>
> 可以变形为**[ ch, h*w]**和**[ h*w, ch]**的矩阵
>
> 再对两个作**内积**得到Gram Matrices

### 理解

> 格拉姆矩阵可以看做feature之间的偏心协方差矩阵（即没有减去均值的协方差矩阵）
>
> 在feature map中，每个数字都来自于一个特定滤波器在特定位置的卷积，因此**每个数字代表一个特征的强度**
>
> Gram计算的实际上是**两两特征之间的相关性**，哪两个特征是同时出现的，哪两个是此消彼长的等等。
>
> 因为为乘法操作 两两特征同时为高 结果才高

> 格拉姆矩阵用于度量**各个维度自己的特性**以及**各个维度之间的关系**
>
> 内积之后得到的多尺度矩阵中:
>
> - 对角线元素提供了**不同特征图各自的信息**
>
> - 其余元素提供了**不同特征图之间的相关信息**。这样一个矩阵，既能体现出有哪些特征，又能体现出不同特征间的紧密程度

> gram矩阵是计算每个通道 i 的feature map与每个通道 j 的feature map的内积
>
> gram matrix的每个值可以说是代表 **I 通道的feature map与 j 通道的feature map的互相关程度**

## 参考链接

- https://www.cnblogs.com/yifanrensheng/p/12862174.html
- https://blog.csdn.net/weixin_40759186/article/details/87804316
- https://www.cnblogs.com/subic/p/8110478.html
