---
title: 有趣的TensorSlow(下)
categories: [AI]
tags: [TensorSlow]
date: 2019-06-02 22:37:36
displayCopyright: true
katex: true
summary: 上篇了解了如何用TensorSlow构建计算图并完成前向传播，这里将继续了解计算损失和利用反向传播更新模型参数。
---



上篇了解了如何用TensorSlow构建计算图并完成前向传播，这里将继续了解计算损失和利用反向传播更新模型参数。

## TensorSlow简介

TensorSlow是极简的模仿TensorFlow的API的python包。

>本文是对原作者danielsabinasz的[教程](http://www.deepideas.net/deep-learning-from-scratch-theory-and-implementation/)的基础上，添加了一点自己的理解。
>
>TensorSlow [**Github repo地址**](https://github.com/danielsabinasz/TensorSlow)
>TensorSlow原作者[**英文教程**](http://www.deepideas.net/deep-learning-from-scratch-theory-and-implementation/)
>
>*The source code has been built with maximal understandability in mind, rather than maximal efficiency.*





## Perceptron Example

本文中将会使用TensorSlow处理稍微复杂一点的模型。首先利用上篇的代码，搭建一个表征Perceptron的计算图。后面将以这个Perceptron为例，介绍TensorSlow如何进行损失计算和反向传播的。值得一提的是，这里的输入和参数已经是向量。




![Perceptron计算图](https://pic.downk.cc/item/5e4d321648b86553eea745db.png)

```python
Graph().as_default()
X = placeholder()

# Create a weight matrix for 2 output classes:
# One with a weight vector (1, 1) for blue and 
# one with a weight vector (-1, -1) for red
W = Variable([
    [1, -1],
    [1, -1]
])
b = Variable([0, 0])
p = softmax( add(matmul(X, W), b) )

session = Session()
output_probabilities = session.run(p, {
    X: np.concatenate((blue_points, red_points))
})
```

## 反向传播

上面构建的计算图，能够实现了Perceptron模型的前向传播。目前为止，模型的参数都是初值，已经给定的。这些仅仅为初值的参数效果好吗？那么需要用**损失函数**来评价当前参数。

### 损失函数

对于二分类问题而言，最大似然估计(MLE)通常是比较好的参数估计的选择。MLE的损失函数形式是负对数似然，也就是交叉熵:

$$J = - \sum_{i=1}^N \sum_{j=1}^C c_{i, j} \cdot log  p_{i, j}$$

其中，$p_{ij}$代表模型对第$i$个样本预测是第$j$个类别的分数，$c_{ij}$代表第$i$个样本是否是第$j$个类别。同样，损失函数$J$也对应于计算图中的一个operation节点：


<!--
![添加损失op](https://raw.githubusercontent.com/yinyajun/yinyajun.github.io/master/images/figure/loss_graph.png)
-->

![添加损失op](https://pic.downk.cc/item/5e4d321648b86553eea745d7.png)

怎么建立这个operation节点？在原计算图的基础上，可以将节点$J$写成这样：
$$\sum_{i=1}^N \sum_{j=1}^C (c \odot log \, p)_{i, j}$$

其中，$c=[c_{i1},\dots,c_{iC}]$是第$i$个样本的label的one-hot向量。可以发现，这是由多个基本的operation节点组成的。实现下面这些基础的operation节点并加以组合，即可得到节点$J$。



| 节点                | 描述                                                 |
| ------------------- | ---------------------------------------------------- |
| $log$ 节点          | The element-wise logarithm<br> of a matrix or vector |
| $\odot$ 节点        | The element-wise product<br> of two matrices         |
| $\sum_{j=1}^C$ 节点 | Sum over the<br>columns of a matrix                  |
| $\sum_{i=1}^N$ 节点 | Sum over the<br>rows of a matrix                     |
| $-$ 节点            | Taking the negative                                  |

基础的operation节点写法在上篇已有介绍，这里仅以log节点为例，代码如下：

```python
class log(Operation):
    """Computes the natural logarithm of x element-wise.
    """
    def __init__(self, x):
        super().__init__([x])
    
    def compute(self, x_value):
        return np.log(x_value)
```
最后，组合这些operation节点，可以完整的给出节点$J$并计算出loss：
```python
# Create a new graph
Graph().as_default()

X = placeholder()
c = placeholder()

W = Variable([
    [1, -1],
    [1, -1]
])
b = Variable([0, 0])
p = softmax( add(matmul(X, W), b) )

# Cross-entropy loss
J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))

session = Session()
print(session.run(J, {
    X: np.concatenate((blue_points, red_points)),
    c:
        [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)
    
}))
```

### 梯度下降

上面的步骤已经计算出模型损失函数。损失函数越小，意味着该模型参数下，模型输出和真实标签越接近。而搜索模型参数，使损失函数最小的方法叫做**梯度下降**，流程如下

>1. 参数$W$和$b$设置随机初始值。
>1. 计算$J$对$W$和$b$的梯度。
>1. 分别在其负梯度的方向上下降一小步(由`learning_rate`控制步长)。
>1. 回到step 2，继续执行，直到收敛。


而`GradientDescentOptimizer`就实现了上述流程中的step 3：

```python 
class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def minimize(self, loss):
        learning_rate = self.learning_rate
        class MinimizationOperation(Operation):
            def compute(self):
                grad_table = compute_gradients(loss)
                
                # Iterate all variables
                for node in grad_table:
                    if type(node) == Variable:
                        grad = grad_table[node]            
                        # Take a step along the direction of the negative gradient
                        node.value -= learning_rate * grad
        return MinimizationOperation()
```
上述代码中`compute_gradients`函数对应于step 2，将在下一个小节介绍。`grad_table`这个字典存放了损失函数$J$节点对图中所有Variable节点的当前梯度（因为只有Variable节点才需要更新）。



### 梯度计算

梯度的计算根据导数的**链式法则**。

#### 链式法则

链式法则是微积分中的求导法则，用于求一个复合函数的导数。示例如下：

<!--
![链式法则](https://raw.githubusercontent.com/yinyajun/yinyajun.github.io/master/images/figure/abcde2.png)
-->

![链式法则](https://pic.downk.cc/item/5e4d311f48b86553eea6f991.png)
$$
\begin{aligned}
\frac{\partial e}{\partial a}
&= \frac{\partial e}{\partial d} \cdot \frac{\partial d}{\partial a}\\\\
&= \frac{\partial e}{\partial d} \cdot \left( \frac{\partial d}{\partial b} \cdot \frac{\partial b}{\partial a} + \frac{\partial d}{\partial c} \cdot \frac{\partial c}{\partial a} \right)\\\\
&= \frac{\partial e}{\partial d} \cdot \frac{\partial d}{\partial b} \cdot \frac{\partial b}{\partial a} + \frac{\partial e}{\partial d} \cdot \frac{\partial d}{\partial c} \cdot \frac{\partial c}{\partial a}
\end{aligned}
$$

### 梯度计算

通过上面的链式法则，计算损失函数节点（记为loss节点）对当前节点$n$的梯度，也是链式的：

1. 计算loss节点对当前$n$节点的consumer节点的输出的梯度$G$.
2. 计算$n$节点的consumer节点对$n$节点梯度$\times G$，将所有consumer节点计算出的梯度全部相加。

```
# 计算当前节点梯度的伪代码
grad_n = grad_child1 * grad_child1_n + ... 
        + grad_childk * grad_childk_n
```

* `grad_n` : $\partial\text{loss}/\partial{n}$
* `chlid1`: a child node of node $n$
* `grad_child1`: $\partial\text{loss}/\partial{\text{child1}}$
* `grad_child1_n`: $\partial\text{child1}/\partial{n}$


先计算子节点梯度，然后得到当前节点梯度，完全按照链式法则。由于当前节点的梯度计算依赖子节点的梯度计算，因此可以使用拓扑排序。而作者使用了BFS来完成拓扑排序。从反图角度而言，每个节点的入度都为0，可以放心使用BFS来完成拓扑排序。

通过一步步迭代，能够得到所有节点的梯度。其中$\partial\text{child1}/\partial{n}$，也就是子节点的输出对于该节点的输出的梯度如何计算？这个工作应该由子节点来完成。

对于一个operation节点，提前定义这个operation的梯度计算函数，并使用装饰器`@RegisterGradient`来将实现了计算梯度函数的opeartion节点注册到全局变量`_gradient_registry`中，具体细节见下一小节。整体的梯度计算过程如下：

```python
from queue import Queue
def compute_gradients(loss):
    grad_table = {}
    grad_table[loss] = 1
    
    # Perform a breadth-first search, backwards from the loss
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)
    
    while not queue.empty():
        node = queue.get()
        if node != loss:
            grad_table[node] = 0
            # Iterate all consumers
            for consumer in node.consumers:
                lossgrad_wrt_consumer_output = grad_table[consumer]
                # Retrieve the function which computes gradients with respect to
                # consumer's inputs given gradients with respect to consumer's output.
                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]
                # Get the gradient of the loss with respect to all of consumer's inputs
                lossgrads_wrt_consumer_inputs = bprop(consumer, lossgrad_wrt_consumer_output)
                
                if len(consumer.input_nodes) == 1:
                    # If there is a single input node to the consumer, lossgrads_wrt_consumer_inputs is a scalar
                    grad_table[node] += lossgrads_wrt_consumer_inputs
                else:
                    # Otherwise, lossgrads_wrt_consumer_inputs is an array of gradients for each input node
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)
                    # Get the gradient of the loss with respect to node
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]
                    grad_table[node] += lossgrad_wrt_node
    
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)
            
    return grad_table
```

其中一些细节还需要自己体会代码。

### Operation节点的梯度
根据operation操作的`compute`方法给出节点正向传播的函数，据此计算梯度函数，然后注册到全局变量中即可。

以矩阵乘法`matmul`为例：给定对于$AB$的梯度$G$，其对于$A$的梯度是$GB^T$，对于$B$的梯度是$A^TG$，所以该节点的梯度是一个向量。

```python
@RegisterGradient("matmul")
def _matmul_gradient(op, grad):
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad.dot(B.T), A.T.dot(grad)]
```
同样的`sigmoid`的梯度可以写为$G \cdot \sigma(a) \cdot \sigma(1-a)$
```python
@RegisterGradient("sigmoid")
def _sigmoid_gradient(op, grad):
    sigmoid = op.output
    return grad * sigmoid * (1-sigmoid)
```

## Perceptron Train

完整的Perceptron模型训练代码如下：

```python
# Create a new graph
Graph().as_default()

X = placeholder()
c = placeholder()

# Initialize weights randomly: step 1
W = Variable(np.random.randn(2, 2))
b = Variable(np.random.randn(2))

p = softmax( add(matmul(X, W), b) )
J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))

# step 2 and step 3
minimization_op = GradientDescentOptimizer(learning_rate = 0.01).minimize(J)

feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    c:  [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)
}

session = Session()
# Perform 100 gradient descent steps, step 4
for step in range(100):
    J_value = session.run(J, feed_dict)
    if step % 10 == 0:
        print("Step:", step, " Loss:", J_value)
    session.run(minimization_op, feed_dict)

W_value = session.run(W)
print("Weight matrix:\n", W_value)
b_value = session.run(b)
print("Bias:\n", b_value)
```
	Step: 0  Loss: 202.782788396
	Step: 10  Loss: 4.04566479054
	Step: 20  Loss: 2.69644468305
	Step: 30  Loss: 2.00506261735
	Step: 40  Loss: 1.62202006027
	Step: 50  Loss: 1.39268559111
	Step: 60  Loss: 1.24498439759
	Step: 70  Loss: 1.14348265257
	Step: 80  Loss: 1.06965484385
	Step: 90  Loss: 1.01324253829
	Weight matrix:
	 [[ 1.27496197 -1.77251219]
	 [ 1.11820232 -2.01586474]]
	Bias:
	 [-0.45274057 -0.39071841]

可以发现loss的确是不断降低的，模型参数不断优化更新。

## Multi-Layer Perceptrons

使用TensorSlow来处理更加复杂的问题：分类的决策边界更加复杂。这里也搭建了更加复杂的模型MLP。

### 数据分布

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorslow as ts

# Create two clusters of red points centered at (0, 0) and (1, 1), respectively.
red_points = np.concatenate((
    0.2 * np.random.randn(25, 2) + np.array([[0, 0]] * 25),
    0.2 * np.random.randn(25, 2) + np.array([[1, 1]] * 25)
))

# Create two clusters of blue points centered at (0, 1) and (1, 0), respectively.
blue_points = np.concatenate((
    0.2 * np.random.randn(25, 2) + np.array([[0, 1]] * 25),
    0.2 * np.random.randn(25, 2) + np.array([[1, 0]] * 25)
))

# Plot them
plt.scatter(red_points[:, 0], red_points[:, 1], color='red')
plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue')
plt.show()
```
![真实数据分布](https://pic.downk.cc/item/5e4d393a48b86553eea9d311.png)
### 计算图

![MLP计算图](https://pic.downk.cc/item/5e4d393a48b86553eea9d313.png)

```python
# Create a new graph
ts.Graph().as_default()

# Create training input placeholder
X = ts.placeholder()
# Create placeholder for the training classes
c = ts.placeholder()

# Build a hidden layer
W_hidden1 = ts.Variable(np.random.randn(2, 4))
b_hidden1 = ts.Variable(np.random.randn(4))
p_hidden1 = ts.sigmoid(ts.add(ts.matmul(X, W_hidden1), b_hidden1))

# Build a hidden layer
W_hidden2 = ts.Variable(np.random.randn(4, 8))
b_hidden2 = ts.Variable(np.random.randn(8))
p_hidden2 = ts.sigmoid(ts.add(ts.matmul(p_hidden1, W_hidden2), b_hidden2))

# Build a hidden layer
W_hidden3 = ts.Variable(np.random.randn(8, 2))
b_hidden3 = ts.Variable(np.random.randn(2))
p_hidden3 = ts.sigmoid(ts.add(ts.matmul(p_hidden2, W_hidden3), b_hidden3))

# Build the output layer
W_output = ts.Variable(np.random.randn(2, 2))
b_output = ts.Variable(np.random.randn(2))
p_output = ts.softmax(ts.add(ts.matmul(p_hidden3, W_output), b_output))

# Build cross-entropy loss
J = ts.negative(ts.reduce_sum(
    ts.reduce_sum(ts.multiply(c, ts.log(p_output)), axis=1)))

# Build minimization op
minimization_op = ts.train.GradientDescentOptimizer(learning_rate=0.03).minimize(J)

# Build placeholder inputs
feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    c:  [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)
}

# Create session
session = ts.Session()

# Perform 100 gradient descent steps
for step in range(2000):
    J_value = session.run(J, feed_dict)
    if step % 100 == 0:
        print("Step:", step, " Loss:", J_value)
    session.run(minimization_op, feed_dict)
```
	Step: 0  Loss: 105.25316761015766
	Step: 100  Loss: 54.82276887616324
	Step: 200  Loss: 18.531741905961816
	Step: 300  Loss: 10.88319073941583
	Step: 400  Loss: 5.167908735173651
	Step: 500  Loss: 4.015258056948531
	...
	Step: 1400  Loss: 0.1973992659737359
	Step: 1500  Loss: 0.17700496511514013
	Step: 1600  Loss: 0.1602628699213534
	Step: 1700  Loss: 0.14629349101006833
	Step: 1800  Loss: 0.13447502395360536
	Step: 1900  Loss: 0.1243562427509077

### 可视化

```python
# Visualize classification boundary
xs = np.linspace(-2, 2)
ys = np.linspace(-2, 2)
pred_classes = []
for x in xs:
    for y in ys:
        pred_class = session.run(p_output,
                                 feed_dict={X: [[x, y]]})[0]
        pred_classes.append((x, y, pred_class.argmax()))
xs_p, ys_p = [], []
xs_n, ys_n = [], []
for x, y, c in pred_classes:
    if c == 0:
        xs_n.append(x)
        ys_n.append(y)
    else:
        xs_p.append(x)
        ys_p.append(y)
plt.plot(xs_p, ys_p, 'ro', xs_n, ys_n, 'bo')
plt.show()
```
![决策边界](https://pic.downk.cc/item/5e4d393a48b86553eea9d30f.png)

## 小结

短小精悍的TensorSlow揭示了了深度模型框架的基本工作机理。当然，它只适用于教学，生产环境下的深度学习框架软件将更加复杂。



