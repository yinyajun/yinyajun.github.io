---
title: 有趣的TensorSlow(上)
copyright: true
categories: [AI]
tags: [TensorSlow]
date: 2019-06-01 22:37:36
katex: true
summary: TensorFlow的许多概念，如graph, session, operation等，为什么要这么设计？Github上的TensorSlow用纯python来模仿了TF的底层api，加深理解TF中的底层概念。
---

TensorFlow的许多概念，如graph, session, operation等，为什么要这么设计？Github上的TensorSlow用纯python来模仿了TF的底层api，加深理解TF中的底层概念。

## TensorSlow简介

* 极简的模仿TensorFlow的API的python包。
* 使用纯python作为后端。
* 仅用作教学目的，帮助理解TensorFlow底层原理。
* 代码量非常少，跟着教程走，很容易看完。

> 本文是对原作者danielsabinasz的[教程](http://www.deepideas.net/deep-learning-from-scratch-theory-and-implementation/)
> 的基础上，添加了一点自己的理解。
>
>
>
>TensorSlow [**Github repo地址**](https://github.com/danielsabinasz/TensorSlow)
> TensorSlow原作者[**英文教程**](http://www.deepideas.net/deep-learning-from-scratch-theory-and-implementation/)
>
>*The source code has been built with maximal understandability in mind, rather than maximal efficiency.*

## 计算图Computational Graphs

计算图是一种有向图，它是以图的形式来表示或计算数学函数。和普通的图一样，计算图中也有节点和边。

* **节点**：要么是提供输入数据的节点，要么是代表操作数据的函数的节点。
* **边**：函数参数（或者说数据依赖），以流的形式为节点传输数据。
* **Tensor**: 节点的输入和输出数据，其实就是一个多维的array。

下图展示了一个计算图，它描述了怎么计算输入节点$x$和$y$的和$z$的过程。

![计算图](https://pic.downk.cc/item/5e4d3a8248b86553eeaa37eb.png)

$x$和$y$都是$z$的输入节点，而$z$是$x$和$y$的消费节点。节点$z$描述了这么一个方程：

$$z:\mathcal{R}^2 \rightarrow \mathcal{R}, z(x,y) = x + y.$$

计算图这个概念非常有用，特别当计算非常复杂的时候，下面的例子对应于一个仿射变换:
$$z(A,x,b)= Ax+b$$

![仿射变换的计算图](https://pic.downk.cc/item/5e4d3a8248b86553eeaa37ed.png)

首先了解各类型节点的表示，从节点输入，节点输出和节点操作来考察各类型节点。

### Operations节点

每个operation节点以下面三个表征：

* **节点操作**：当operation节点的输入的值给定时，用`compute`函数来计算该operation节点的输出
* **节点输入**：`input_nodes`列表，可以是varibles节点或者是其他operations节点
* **节点输出**：`consumers`列表，它们将该operation节点的输出作为它们的输入。

上面三个含义都是显而易见的描述operation节点的操作，在`Opearation`类中，用三个成员表示

```python
class Operation:
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes

        # Initialize list of consumers 
        self.consumers = []

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)

        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)

    def compute(self):
        pass
```

`compute`方法是需要每个operation节点子类去实现。

```python
# Addition Operation节点示例
class add(Operation):
    """Returns x + y element-wise.
    """

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        self.inputs = [x_value, y_value]
        return x_value + y_value
```

```python
# Matrix Multiplicaiton Operation节点示例
class matmul(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        self.inputs = [a_value, b_value]
        return a_value.dot(b_value)
```

### Placeholder节点

计算图中为了计算输出，必须要向图中提供一次输入数据。而Placeholder节点，正如其名，就是用来干这事的。在仿射变换计算图的例子中，$x$就是这种节点。

- **节点操作**：无
- **节点输入**：无
- **节点输出**：`consumers`列表，它们将该节点的输出作为它们的输入。

```python
class placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self):
        self.consumers = []
        _default_graph.placeholders.append(self)
```

### Variables节点

在仿射变换的例子中，$x$,$A$和$b$都不是operation节点，但是$x$与$A$和$b$有一些区别，$x$是纯粹的输入placeholder节点，而$A$和$b$是能不断变更输出值，它们是Variable节点。这些Variable节点虽然没有输入，但本身有初值。Variable节点是计算图的固有成分。

- **节点操作**：无
- **节点输入**：无
- **节点输出**：`consumers`列表，它们将该节点的输出作为它们的输入。

```python
class Variable:
    """Represents a variable (i.e. an intrinsic, changeable parameter of a computational graph).
    """

    def __init__(self, initial_value=None):
        self.value = initial_value
        self.consumers = []

        _default_graph.variables.append(self)
```

### The Graph class

使用`Graph`来绑定所有创建的节点。当创建新的graph的时候，可以调用`as_default`方法来设置默认图`_default_graph`
，这样我们不用显示地去将节点绑定到图中。

```python
class Graph:
    def __init__(self):
        """Construct Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self
```

### Example

通过已经建立的类，来建立仿射变换例子的计算图：

$$
z = \begin{pmatrix}
1 & 0 \\\\
0 & -1
\end{pmatrix}
\cdot
x

+

\begin{pmatrix}
1 \\\\
1
\end{pmatrix}
$$

```python
# Create a new graph
Graph().as_default()

# Create variables
A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])

# Create placeholder
x = placeholder()

# Create hidden node y
y = matmul(A, x)

# Create output node z
z = add(y, b)
```

### Session

建立完计算图，那么开始思考，如何计算出节点的输出？输出节点通常是operation节点。为了正确计算输出节点的输出，需要按正确的顺序计算。仍以仿射变换为例，要计算$z$必须先计算出中间结果$y$。也就是说，
**必须保证节点是按顺序执行的，计算节点$o$之前，节点$o$的所有输入节点已经完成计算**，使用**拓扑排序**即可满足要求。

拓扑排序，是原图的reverse post order，和反图的post order一致。这里使用反图的post order来得到拓扑顺序。这一系列计算都封装在了Session中。

```python
import numpy as np


class Session:
    """Represents a particular execution of a computational graph.
    """

    def run(self, operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)
        # Iterate all nodes to determine their value
        for node in nodes_postorder:
            if type(node) == placeholder:
                # Set the node value to the placeholder value from feed_dict
                node.output = feed_dict[node]
            elif type(node) == Variable:
                # Set the node value to the variable's value attribute
                node.output = node.value
            else:  # Operation
                # Get the input values for this operation from node_values
                node.inputs = [input_node.output for input_node in node.input_nodes]
                # Compute the output of this operation
                node.output = node.compute(*node.inputs)

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)
        # Return the requested node value
        return operation.output


def traverse_postorder(operation):
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder
```

可见，`run`方法对要计算的operation节点进行了一次拓扑排序，按照这个顺序，依次计算节点。

#### Example

完成仿射变换例子的输出.

$$
z=\begin{pmatrix}1 & 0 \\\\ 0 & -1 \end{pmatrix} \cdot \begin{pmatrix}1 \\\\ 2\end{pmatrix} + \begin{pmatrix}
1 \\\\
1
\end{pmatrix}=
\begin{pmatrix}
2 \\\\
-1
\end{pmatrix}
$$


```python
session = Session()
output = session.run(z, {
    x: [1, 2]
})
print(output)
```

	[ 2 -1]

## 小结

目前，已经可以搭建计算图，用来计算一些复杂函数了。如果用计算图来搭建神经网络，目前的代码完全能够完成网络的前向传播。[下篇](https://yinyajun.github.io/infomation-tech/tensorslow-02/)
将会涉及到loss计算及反向传播在计算图中如何实现。
