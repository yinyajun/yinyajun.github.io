---
title: 赵世钰《强化学习》复习笔记01
copyright: true
categories: [强化学习]
tags: [西湖大学,赵世钰]
date: 2023-07-15 15:37:36
katex: true
summary: 我的强化学习入门课程就是赵世钰老师的[强化学习](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)，二刷后记录一些复习笔记。这篇是1-3章的复习笔记。
---

我的强化学习入门课程就是赵世钰老师的[强化学习](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)，二刷后记录一些复习笔记。这篇是1-3章的复习笔记。

## 第一章

### 基础概念


| 概念              | 解释                                         |
| ----------------- | -------------------------------------------- |
| State & Action    |                                              |
| State Transition  | 从一个state跳转到另一个state的条件概率       |
| Reward            | state下采取某一action后环境给的奖励          |
| Policy            | 根据state采取action的条件概率                |
| Trajectory        | 从某个state开始的state-action-reward chain   |
| Return            | Trajectory中的所有reward总和                 |
| Discounted Return | 对于无限长的Trajectory，可以得到收敛的Return |
| Episode/Trial     | agent会停在terminal state的trajectory        |

有了这些基础概念，那么可以来构建马尔可夫决策过程。



### 马尔可夫决策过程（MDP）

MDP是在马尔可夫过程（MP）的基础上，MP有状态转移+马尔可夫性质。

* 在状态跳转后加上reward，就到了马尔可夫奖励过程。
* 状态跳转前再加上action，就到了MDP。

MDP就是强化学习的数学框架。那么MDP有哪些核心要素？

* **State space**：$\mathcal{S}$

* **Action space**: $\mathcal{A(s)}$

* **Reward set**: ${\mathcal{R(s,a)}}$

* **State transition prob**: $p(s'|s,a)$

* **Reward prob**: $p(r|s,a)$

* **Policy**: $\pi(a|s)$

* **Markov Property**（MDP version）

  next state or reward **only** depends on the current state and action.

有了这些核心要素，那么MDP形式上就是

$$S_1 \rightarrow A_1 \rightarrow R_2 \rightarrow S_2 \rightarrow A_2 \rightarrow R_3 \dots$$

需要找到一个Policy，来最大化这个Return。

> 如果采取固定的policy，那么这里的状态转移就退化为MP的状态转移，MDP也退化为MP。



### 环境模型

在MDP中，$p(s'|s,a)$和$p(r|s,a)$是由环境决定的model（dynamics）。

它们不是我们求解的目标，但是也会影响整个MDP。

* 如果环境model已知（白盒环境），求解方法为Model-based.
* 如果环境model未知（黑盒环境），求解方法为Model-free.



## 第二章

这章只有一个东西，就是state value。然后围绕state value的求解，引出了Bellman方程。

> 首先，好的Policy是怎么eval的？用Return，特别是discounted return。

怎么计算return？

1. Definition: $$v_1 = r_1 + \gamma r_2 + \gamma^2 r_3 + ...$$
2. Bootstrapping: $$v_1=r_1 + \gamma(r_2 + \gamma r_3 + ...) = r_1 + \gamma v_2$$

bootstrapping的方式乍看很绕，其实就是递归的定义。具体而言，从一个状态出发得到的return，依赖其他状态出发得到的return。



对于所有的状态，都有这么一个方程，联立所有方程，可以得到矩阵形式

$$v = r + \gamma P v$$

其实这就是Bellman方程，当然需要进一步引入随机变量的版本。

### State Values

更正式的提出了trajectory的discounted return

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$$

这是个随机变量，因为所有的R都是随机变量。为了避免return的随机性对估计policy的影响，引入了state value。

$$v_\pi(s) = \mathbb{E}[G_t| S_t=s]$$

它的定义很直接，就是return的期望。至此，我们将使用state value代替return来eval policy。



### Bellman方程

同样引入递归定义


$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...\\\\ 
& = R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + ...)\\\\ 
& = R_{t+1} + \gamma G_{t+1}
\end{aligned}
$$


含义很自然，这一时刻的return = 当前reward + 下一时刻的return.

因此，state value的可以拆为
$$v_\pi(s)  = \mathbb{E} [R_{t+1} | S_t = s] + \gamma \mathbb{E} [ G_{t+1} | S_t = s]$$

* 前者是immediate reward。

  是当前状态所能获取的reward的期望。

* 后者是future rewards。

  当前状态开始，跳到下一时刻的某个状态后的return的期望。

#### immediate reward

$$\mathbb{E}[R_{t+1}| S_t=s] = \sum_{a} \pi(a|s) \sum_r p(r|s,a)r$$

随机性在于policy和环境奖励的随机。

#### future rewards

$$
\begin{aligned}
&\mathbb{E}[G_{t+1}| S_t = s]  \\\\
&= \sum_{s'} \mathbb{E}[G_{t+1}| S_{t+1}=s'] p(s'|s) \\\\
&= \sum_{s'}v_\pi (s') p(s'|s)
\end{aligned}
$$

这里比较跳的步骤是用了markov的性质，Return只和当前时刻的State有关，和前一个时刻的State无关。



随机性在于状态转移的随机。

进一步，将MP形式的状态转移可以展开为MDP形式的状态转移

$$p(s'|s) = \sum_a p(s'|s,a) \pi(a|s).$$



#### 小说明

这章在引入了两次递归定义，一次是在Return的bootstrapping求解，一次在State value的定义。

初学的时候，觉得有点混淆，因为这两个好像是同一个东西。它们的区别在于

* 前者是实现，用小写字母表示。某个state的return依赖其他state的return。
* 后者是随机变量，用大写字母表示。某个时刻state的return依赖未来时刻的return。
* 因为Return是随机变量，当具体采样到某条trajectory时，某个时刻的state就是其实现，为一个具体的state。

#### Bellman Equation

综上可以得到elementwise形式的bellman方程，

$$
\begin{aligned}
v_\pi(s)  &= \mathbb{E} [R_{t+1} | S_t = s] + \gamma \mathbb{E} [ G_{t+1} | S_t = s] \\\\
& = \sum_{a} \pi(a|s)\sum_R p(r|s,a)r + \gamma \sum_a \pi(a|s) \sum_{s'} p(s' |s,a)v_\pi(s') \\\\
& = \sum_a \pi(a|s) \left[\sum_r p(r|s,a)r + \gamma\sum_{s'}p(s'|s,a)v_{\pi}(s')\right], \text{     for all  } s \in \mathcal{S}
\end{aligned}
$$
其中，

* $v_\pi(s)$和$v_\pi(s')$待求解。

* $\pi(a|s)$是当前policy。

* $p(r|s,a)$和$p(s'|s,a)$是环境model，白盒环境中已知。

  

一个简单的变体是
$$v_\pi(s) = \sum_a \pi(a|s) \sum_{s'}\sum_r p(s',r | s,a)[r + \gamma v_\pi(s')].$$

#### matrix-vector form

重写bellman方程

$$v_\pi(s) = r_\pi(s) + \gamma\sum_{s'} p_\pi(s'|s) v_\pi(s')$$

其中，

$$r_\pi(s) = \sum_a \pi(a|s) \sum_r p(r|s,a) r,$$

$$p_\pi(s'|s) = \sum_a \pi(a|s)p(s'|s,a),$$

将所有state的方程联立，可得

$$v_\pi = r_\pi + \gamma P_\pi v_\pi.$$

### Bellman equation solution

#### closed-form solution

$$v_\pi = (I - \gamma P_\pi)^{-1} r_\pi.$$

* 可逆性，通过圆盘定理估计出所有eigenvalue都大于0，所以可逆。
* $(I - \gamma P_\pi)^{-1}  = I + \gamma P_\pi + \gamma^2 P_\pi ^ 2 + \dots$
  * 可逆矩阵本就有无限项的含义。
  * 可逆矩阵太难求了。

#### Iterative solution

$$v_{k+1}=  r_\pi + \gamma P_\pi v_k, \ \ \ \ k=0,1,2,...$$ 

注意此时上式并不满足bellman方程。通过迭代生成一系列$\{v_0, v_1, v_2, ...\}$，当$k$足够大的时候，$v_k$才是满足bellman方程的解。

$$v_k = v_\pi = (I  - \gamma P_\pi)^{-1} r_\pi, \ \text{as  } k \rightarrow \infty$$

证明方法是$\delta_k = v_k - v_\pi$随着迭代步数增加而趋于0。

### Action Value

有了State value的基础，可以介绍Action value了。

$$q_\pi(s,a) = \mathbb{E}[G_t| S_t =s , A_t =a].$$

定义和state value很像，只是依赖于state-action pair.

#### Action Value -> State Value
State value是某一state下所有可能Action value的期望。

$$v_\pi(s) = \sum_a \pi(a|s) q_\pi(s,a)$$


#### State Value -> Action Value

回顾$v_\pi(s)$的展开式

$$v_\pi(s) = \sum_a \pi(a|s) \left[\sum_r p(r|s,a)r + \gamma\sum_{s'}p(s'|s,a)v_{\pi}(s')\right]$$

可得Action value的展开式

$$q_\pi(s,a) =\sum_r p(r|s,a)r + \gamma\sum_{s'}p(s'|s,a)v_{\pi}(s')$$

#### Bellman Equation in terms of Action Value

通过上面的式子，可以移除掉$v_\pi(s)$

$$q_\pi(s,a) = \sum_r p(r|s,a) r + \gamma \sum_{s'}p(s'|s,a)\sum_{a'} \pi(a'|s')q_\pi(s',a')$$

矩阵形式稍微有些麻烦，不方便打，直接看书。矩阵的维度，对于理解上式非常有帮助。

## 第三章

这章围绕optimal state value.



### Optimal Policy

相比于其他policy，optimal policy在所有state上的state value都是最大的，称之为$\pi ^*.$ 我们需要探讨这三点

* Existence
* Uniqueness
* Algorithm



### Bellman optimality equation (BOE)

$$v(s) = \max_{\pi}\sum _ a \pi(a|s) q(s,a)$$

由于$q(s,a)$中包含了$v(s)$，所以右边求最值的式子中有两个未知量$q(s,a)$和$\pi(a|s)$.

先求optimal policy $\pi^*,$

$$
\pi^*(a|s) = 
\begin{cases} 
1 & \text{if } a = a^* \\\\
0 & \text{if } a \neq a^*
\end{cases}
$$

其中，$a^* = \arg \max_a q(s, a).$ Optimal policy就是找action value最大的动作。

所以，方程的左右边只有$v$，用矩阵形式可写为

$$v = \max_\pi(r_\pi+ \gamma P_\pi v)$$

$\pi$在max的作用下将不在是变量，只有$v$是变量。所以BOE相当于求解$v= f(v)$.

> Contraction mapping theorem
>
> 1. 存在不动点$x^*,$ 使得$$f(x^*)=x^*$
> 2. 只要满足，$|| f(x_1)-f(x_2)||\leq \gamma ||x_1 -x_2||$
>
> 定理可以保证：
>
> 1. 存在性
> 2. 唯一性
> 3. 使用迭代算法来求解不动点，以指数速率收敛到不动点
>
> 证明思路：构建Cauchy序列

有了Contraction mapping theorem，可证BOE的右侧的函数映射是contraction mapping（看书）

可以不加证明的看，

* max去除$\pi$的影响
* $r_\pi$和$v$无关
* $P_\pi$是stochastic matrix，作用下不会超过$v$的最大元素

大概可以认为是$||f(v_1)-f(v_2)||_\infty \leq\gamma ||v1-v2||_\infty$. 这样，BOE的右侧函数就是一个Contraction mapping.

### Solving BOE

BOE就可以利用Contraction mapping theorem来求得optimal state values. 通过下面公式迭代

$$v_{k+1} = \max_\pi (r_\pi + \gamma P_\pi v_k), \ \ \ k = 0,1,2,...$$

这样的迭代方法称为value iteration.