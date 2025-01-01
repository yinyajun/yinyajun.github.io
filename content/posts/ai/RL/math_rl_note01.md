---
title: 西湖大学强化学习复习笔记01
copyright: true
categories: [强化学习]
tags: [西湖大学,赵世钰]
date: 2023-07-15 15:37:36
katex: true
summary: 赵世钰老师的强化学习《mathematical foundations of reinforcement learning》前3章复习笔记。
---

我的强化学习入门课程就是赵世钰老师的[强化学习](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)，二刷后记录一些复习笔记。这篇是1-3章的复习笔记。

## 第一章

### 基础概念

* State & Action

* State Transition: $P(s'|s)$

* Policy: $\pi(a|s)$

* Reward: $r(s, a)$

* Trajectory: state-action-reward chain

* Return:  sum of all rewards along the trajectory (can eval policy)

  * Immediate reward 

    采取当前action后得到的reward

  * Future rewards

    离开当前状态，后续得到的所有reward

* Discounted Return （for infinity long trajectories）

* Episode/Trial: agent may stop at some terminal states, the resulting trajectory.

有了这些基础概念，那么可以来构建马尔可夫决策过程。

### 马尔可夫决策过程

MDP过程是在马尔可夫过程上加上reward和action

先列举有哪些组件

* State space：$\mathcal{S}$

* Action space: $\mathcal{A(s)}$

* Reward set: ${\mathcal{R(s,a)}}$

* State transition prob: $p(s'|s,a)$

* Reward prob: $p(r|s,a)$

* Policy: $\pi(a|s)$

* Markov Property

  next state or reward **only** depends on the current state and action.

$p(s'|s,a)$和$p(r|s,a)$是由环境决定的模型。如果Policy固定住了，那么MDP将会退化成MP。

## 第二章

这章围绕state value.

首先，return，特别是discounted return可以用来eval policy，怎么计算return？

1. Definition: $$v_1 = r_1 + \gamma r_2 + \gamma^2 r_3 + ...$$
2. Bootstrapping: $$v_1=r_1 + \gamma(r_2 + \gamma r_3 + ...) = r_1 + \gamma v_2$$

bootstrapping的方式乍看很绕，其实就是递归的定义。

具体而言，从一个状态出发得到的return，依赖其他状态出发得到的return。注意的是，这里不仅仅有一个状态，当计算所有状态的return时，这里就联立了n个方程，对应于n个位置的return，自然可以求得。

矩阵形式更好理解

$$v = r + \gamma P v$$

其实这就是Bellman方程，当然这个形式还无法直接使用。

### State Values

更正式的提出了trajectory的discounted return

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$$

这是个随机变量，因为所有的R都是随机变量。为了避免return的随机性对估计policy的影响，引入了state value。它的定义很直接，就是return的期望。

$$v_\pi(s) = \mathbb{E}[G_t| S_t=s]$$

至此，我们使用state value来eval policy。

### Bellman方程

同样引入递归定义


$$
\begin{align}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...\\ 
& = R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + ...)\\ 
& = R_{t+1} + \gamma G_{t+1}
\end{align}
$$

含义很自然，这一时刻的return=当前reward+下一时刻的return。

state value的定义可以拆为
$$v_\pi(s)  = \mathbb{E} [R_{t+1} | S_t = s] + \gamma \mathbb{E} [ G_{t+1} | S_t = s]$$

* 前者是immediate reward。

  是当前状态所能获取的reward的期望。

* 后者是future rewards。

  当前状态开始，跳到下一时刻的某个状态后的return的期望。

具体怎么展开这两个期望

$$\mathbb{E}[R_{t+1}| S_t=s] = \Sigma_{a} \pi(a|s) \Sigma _r p(r|s,a)r$$

随机性由policy和环境奖励的随机引入。

再看看后者

$$\mathbb{E}[G_{t+1}| S_t = s] \\= \Sigma_{s'} \mathbb{E}[G_{t+1}| S_{t+1}=s'] p(s'|s) \\=  \Sigma_{s'}v_\pi (s') p(s'|s)$$

这里比较跳的步骤是用了markov的性质。随机由下一时刻的return本身和状态转移的随机引入。

这里还有一点对齐的是，
