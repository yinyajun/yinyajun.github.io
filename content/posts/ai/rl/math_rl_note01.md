---
title: 赵世钰《强化学习》复习笔记01
copyright: true
categories: [强化学习]
tags: [西湖大学,赵世钰]
date: 2023-07-15 15:37:36
katex: true
summary: 我的强化学习入门课程就是赵世钰老师的[强化学习](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)，二刷后记录一些复习笔记。这篇是1-3章的复习笔记。

---



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
* 状态跳转前在加上action，就到了MDP。

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

其实这就是Bellman方程。现在这个形式不太方便使用。

### State Values

更正式的提出了trajectory的discounted return

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$$

这是个随机变量，因为所有的R都是随机变量。为了避免return的随机性对估计policy的影响，引入了state value。

$$v_\pi(s) = \mathbb{E}[G_t| S_t=s]$$

它的定义很直接，就是return的期望。至此，我们将使用state value代替return来eval policy。



### Bellman方程

同样引入递归定义


$$
\begin{equation}
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...\\ 
& = R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + ...)\\ 
& = R_{t+1} + \gamma G_{t+1}
\end{aligned}
\end{equation}
$$

含义很自然，这一时刻的return = 当前reward + 下一时刻的return。

因此，state value的可以拆为
$$v_\pi(s)  = \mathbb{E} [R_{t+1} | S_t = s] + \gamma \mathbb{E} [ G_{t+1} | S_t = s]$$

* 前者是immediate reward。

  是当前状态所能获取的reward的期望。

* 后者是future rewards。

  当前状态开始，跳到下一时刻的某个状态后的return的期望。

#### immediate reward

$$\mathbb{E}[R_{t+1}| S_t=s] = \Sigma_{a} \pi(a|s) \Sigma _r p(r|s,a)r$$

随机性在于policy和环境奖励的随机。

#### future rewards



$$
\begin{equation}\begin{aligned}&\mathbb{E}[G_{t+1}| S_t = s]  \\&= \Sigma_{s'} \mathbb{E}[G_{t+1}| S_{t+1}=s'] p(s'|s) \\ &= \Sigma_{s'}v_\pi (s') p(s'|s)\end{aligned}\end{equation}
$$



这里比较跳的步骤是用了markov的性质，Return只和当前时刻的State有关，和前一个时刻的State无关。

随机性在于return的随机和状态转移的随机。



#### 小说明

这章在引入了两次递归定义，一次是在Return的bootstrapping求解，一次在State value的定义。

初学的时候，觉得有点混淆，因为这两个好像是同一个东西。它们的区别在于

* 前者是实现，用小写字母表示。某个state的return依赖其他state的return。
* 后者是随机变量，用大写字母表示。某个时刻state的return依赖未来时刻的return。
* 因为Return是随机变量，当具体采样到某条trajectory时，才是前者。
* 而我们为了更好的衡量Return，采用期望而不是某一个采样。

#### Bellman Equation

综上，

$$v_\pi(s)  = \mathbb{E} [R_{t+1} | S_t = s] + \gamma \mathbb{E} [ G_{t+1} | S_t = s]$$