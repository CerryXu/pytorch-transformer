# pytorch-transformer
Transformer模型的PyTorch实现

学习博主<a href="http://www.jintiankansha.me/t/RcTuLXkjul">这篇文章</a>的代码。

本项目使用
+ python 2.7
+ pytorch 0.4.0

## Transformer架构

![Transformer架构](http://img2.jintiankansha.me/get3?src=http://user-gold-cdn.xitu.io/2018/9/17/165e5814fae0765f?imageView2/0/w/1280/h/960/ignore-error/1)


### 首先，Transformer模型也是使用经典的encoer-decoder架构，由encoder和decoder两部分组成。

上图的左半边用Nx框出来的，就是我们的encoder的一层。encoder一共有6层这样的结构。

上图的右半边用Nx框出来的，就是我们的decoder的一层。decoder一共有6层这样的结构。

输入序列经过word embedding和positional encoding相加后，输入到encoder。

输出序列经过word embedding和positional encoding相加后，输入到decoder。

最后，decoder输出的结果，经过一个线性层，然后计算softmax。

#### Encoder

encoder由6层相同的层组成，每一层分别由两部分组成：

第一部分是一个multi-head self-attention mechanism
第二部分是一个position-wise feed-forward network，是一个全连接层
两个部分，都有一个 残差连接(residual connection)，然后接着一个Layer Normalization。

#### Decoder

和encoder类似，decoder由6个相同的层组成，每一个层包括以下3个部分：

第一个部分是multi-head self-attention mechanism
第二部分是multi-head context-attention mechanism
第三部分是一个position-wise feed-forward network
还是和encoder类似，上面三个部分的每一个部分，都有一个残差连接，后接一个Layer Normalization。

但是，decoder出现了一个新的东西multi-head context-attention mechanism。

