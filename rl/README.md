# 强化学习笔记

## 马尔可夫决策过程 (MDP)

强化学习问题通常建模为**马尔可夫决策过程 (MDP, Markov Decision Process)**，由标准五元组描述：

$$
\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle
$$

### 各符号含义

| 符号 | 名称 | 含义 |
|------|------|------|
| $\mathcal{S}$ | State space（状态空间） | 所有可能状态的集合 |
| $\mathcal{A}$ | Action space（动作空间） | 所有可能动作的集合 |
| $\mathcal{P}$ | Transition probability（状态转移概率） | 在某状态下采取某动作后转移到下一状态的概率 |
| $\mathcal{R}$ | Reward function（奖励函数） | 在某状态下采取某动作所获得的（期望）即时奖励 |
| $\gamma$ | Discount factor（折扣因子） | 衡量未来奖励的重要性，$\gamma \in [0,1]$ |

### 形式化描述

**状态空间** $\mathcal{S}$ 与**动作空间** $\mathcal{A}$ 本质上是同一种东西——都是变量的集合，区别只在于「变量是谁、是离散还是连续」。任意一个状态 $s$ 或动作 $a$ 都可以写成一个 $n$ 维向量：

$$
s = (x_1, x_2, \ldots, x_n) \in \mathcal{S}, \qquad
a = (u_1, u_2, \ldots, u_m) \in \mathcal{A}
$$

每个分量 $x_i$（或 $u_j$）既可以是**离散**的，也可以是**连续**的，于是空间整体表现为离散、连续或两者混合：

$$
\mathcal{S} = \mathcal{X}_1 \times \mathcal{X}_2 \times \cdots \times \mathcal{X}_n, \qquad
\mathcal{X}_i \subseteq \mathbb{Z} \ \text{（离散）} \quad \text{或} \quad \mathcal{X}_i \subseteq \mathbb{R} \ \text{（连续）}
$$

**举例（状态空间 $\mathcal{S}$）**

| 类型 | 例子 | 形式化 |
|------|------|--------|
| 离散 | 网格世界中智能体的格子坐标 | $s = (r, c),\ r,c \in \{0,1,\ldots,N-1\}$ |
| 连续 | 倒立摆的角度与角速度 | $s = (\theta, \dot\theta),\ \theta \in [-\pi, \pi],\ \dot\theta \in \mathbb{R}$ |
| 混合 | 自动驾驶：车道编号(离散)+速度(连续) | $s = (\text{lane}, v),\ \text{lane} \in \{1,2,3\},\ v \in \mathbb{R}_{\ge 0}$ |

**举例（动作空间 $\mathcal{A}$）**

| 类型 | 例子 | 形式化 |
|------|------|--------|
| 离散 | 上下左右四个移动方向 | $a \in \{\uparrow, \downarrow, \leftarrow, \rightarrow\}$ |
| 连续 | 机械臂各关节施加的力矩 | $a = (\tau_1,\ldots,\tau_m),\ \tau_j \in \mathbb{R}$ |
| 混合 | 游戏：选技能(离散)+瞄准角度(连续) | $a = (\text{skill}, \phi),\ \text{skill} \in \{1,\ldots,K\},\ \phi \in [0, 2\pi)$ |

离散 / 连续动作空间的区分直接决定算法选型：离散常用 DQN 一类，连续常用 DDPG、PPO（连续策略）一类。

**状态转移概率** $\mathcal{P}$ 是一个映射 $\mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$：

$$
\mathcal{P}_{ss'}^{a} = \mathbb{P}\left[ S_{t+1} = s' \mid S_t = s,\, A_t = a \right]
$$

满足归一化条件 $\sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^{a} = 1$。这里隐含了**马尔可夫性质**：下一状态只依赖于当前状态和动作，与历史无关。

**奖励函数** $\mathcal{R}$ 是映射 $\mathcal{S} \times \mathcal{A} \to \mathbb{R}$：

$$
\mathcal{R}_s^{a} = \mathbb{E}\left[ R_{t+1} \mid S_t = s,\, A_t = a \right]
$$

其中 $R_{t+1}$ 是智能体在时刻 $t$ 采取动作 $A_t$ 后、环境反馈回来的**即时奖励**（与新状态 $S_{t+1}$ 同时返回，故下标用 $t+1$）。由于环境转移可能是随机的，同样的 $(s,a)$ 每次奖励未必相同，所以取期望 $\mathbb{E}[\cdot]$。

**奖励由环境（问题设计者）定义，而非算法计算。** $R_{t+1}$ 是 MDP 的输入、是任务目标的数学编码；智能体只「接收」奖励，据此估计价值、改进策略，从不计算奖励本身。其底层奖励函数 $r(\cdot)$ 根据需要可以有不同的依赖范围：

| 形式 | 依赖 | 含义 |
|------|------|------|
| $R_{t+1} = r(S_{t+1})$ | 只看当前动作结果 | 只关心到达了什么状态（如到达目标 +1） |
| $R_{t+1} = r(S_t, A_t)$ | 看状态-动作对 | 在某状态做某动作的代价/收益（如施加控制的能耗） |
| $R_{t+1} = r(S_t, A_t, S_{t+1})$ | 看完整状态转移 | 同时依赖起点、动作和落点（最一般的形式） |

> 奖励函数的设计（reward design / shaping）是 RL 工程中最关键也最难的一环：设计不当会导致智能体「钻空子」（reward hacking）。常见做法有稀疏奖励（仅关键事件给分）与稠密/塑形奖励（每步都给引导信号）。

**折扣因子** $\gamma \in [0,1]$ 用于定义累积回报 (return)：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

$\gamma$ 越接近 0 越「短视」（只看眼前奖励），越接近 1 越「远视」（重视长期回报）。

**策略** $\pi$ 定义为状态到动作的（条件）概率分布：

$$
\pi(a \mid s) = \mathbb{P}\left[ A_t = a \mid S_t = s \right]
$$

## 动态规划 (Dynamic Programming)

动态规划假设**完全已知 MDP**，用于 MDP 中的**规划 (planning)**。它解决两类问题：

### Prediction（预测 / 策略评估）

- **输入**：MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$ 和策略 $\pi$
- **或输入**：MRP $\langle \mathcal{S}, \mathcal{P}^{\pi}, \mathcal{R}^{\pi}, \gamma \rangle$
- **输出**：价值函数 $v_\pi$

由于策略 $\pi$ 固定，动作被「吸收」掉，MDP 退化为 **MRP（Markov Reward Process，马尔可夫奖励过程）**——注意它**没有 $\mathcal{A}$**：

$$
\mathcal{P}^{\pi}_{ss'} = \sum_{a \in \mathcal{A}} \pi(a \mid s)\, \mathcal{P}_{ss'}^{a}, \qquad
\mathcal{R}^{\pi}_{s} = \sum_{a \in \mathcal{A}} \pi(a \mid s)\, \mathcal{R}_{s}^{a}
$$

#### 价值函数 (Value Function)

**状态价值函数** $v_\pi(s)$ 定义为：从状态 $s$ 出发、之后一直遵循策略 $\pi$，所能获得的**期望累积回报**：

$$
v_\pi(s) = \mathbb{E}_\pi\left[ G_t \mid S_t = s \right]
= \mathbb{E}_\pi\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s \right]
$$

它衡量「身处状态 $s$ 有多好」。类似地，**动作价值函数** $q_\pi(s,a)$ 衡量「在状态 $s$ 采取动作 $a$、之后遵循 $\pi$ 有多好」：

$$
q_\pi(s,a) = \mathbb{E}_\pi\left[ G_t \mid S_t = s,\, A_t = a \right]
$$

两者关系：对动作按策略求期望即得状态价值 $v_\pi(s) = \sum_{a} \pi(a\mid s)\, q_\pi(s,a)$。

价值函数满足递归的 **Bellman 期望方程**——这正是动态规划求解 $v_\pi$ 的基础：

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a\, v_\pi(s') \right)
$$

写成 MRP（策略已吸收）的紧凑形式：

$$
v_\pi(s) = \mathcal{R}_s^{\pi} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^{\pi}\, v_\pi(s')
$$

> 策略评估 (policy evaluation) 即反复迭代应用上式，使 $v$ 收敛到 $v_\pi$。

#### Reward vs Value

奖励函数和价值函数容易混淆，核心区别如下：

| | Reward 奖励 | Value 价值 |
|---|---|---|
| 看多远 | **单步**（瞬时） | **整个未来**（长期累积） |
| 依赖什么 | 当前转移 $(S_t, A_t, S_{t+1})$ | 当前状态 + 之后一直遵循的策略 $\pi$ |
| 谁决定 | 环境定义，固定不变 | 取决于策略 $\pi$，随策略变化 |
| 数学上 | $R_{t+1}$（一个数 / 其期望 $\mathcal{R}_s^a$） | $v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$（回报的期望） |
| 记号 | $\mathcal{R}_s^a$，不带策略下标 | $v_\pi$，带策略下标 $\pi$ |

**为什么需要 value 而不直接用 reward？** 因为 RL 要的是「长期最优」而非「眼前最优」。Value 计算了**延迟回报 (delayed reward)**：一个动作可能即时 reward 很低，却能进入高价值的状态（比如下棋弃子、先投入再收益），value 把这种未来收益也算进来了。所以 value 衡量的是一整个过程的长期回报，而非单步 reward。

### Control（控制）

- **输入**：MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$（不给定策略）
- **目标**：找到最优策略 $\pi^*$

#### 两类 Bellman 方程

价值函数的递归方程有两种，区别在于**动作如何处理**：期望方程按策略加权求期望，最优方程直接取最大值。

**Bellman Expectation Equation（期望方程）**——针对**给定策略 $\pi$**，动作按 $\pi$ 加权：

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a\, v_\pi(s') \right)
$$

- 回答「遵循策略 $\pi$，这个状态值多少」；**线性**方程，可直接求解。
- 用途：**Prediction（策略评估）**。

**Bellman Optimality Equation（最优方程）**——针对**最优价值函数 $v_*$**，动作取使价值最大的那个：

$$
v_*(s) = \max_{a \in \mathcal{A}} \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a\, v_*(s') \right)
$$

$$
q_*(s,a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a\, \max_{a' \in \mathcal{A}} q_*(s', a')
$$

- 回答「采取最优行为，这个状态最多值多少」；含 $\max$，**非线性**，须迭代求解。
- 用途：**Control（求最优策略）**。

| | Bellman Expectation | Bellman Optimality |
|---|---|---|
| 针对 | 给定策略 $\pi$ | 最优策略 $\pi_*$ |
| 动作如何处理 | 按 $\pi(a\mid s)$ **加权求期望** | **取最大** $\max_a$ |
| 价值对象 | $v_\pi,\ q_\pi$ | $v_*,\ q_*$ |
| 线性性 | 线性（可直接解） | 非线性（含 max，须迭代） |
| 解决的问题 | Prediction（评估策略好坏） | Control（找最优策略） |

#### 三种 DP 算法的对应关系

| Problem | Bellman Equation | Algorithm |
|---|---|---|
| Prediction | Bellman Expectation Equation | Iterative Policy Evaluation |
| Control | Bellman Expectation Equation + Greedy Policy Improvement | Policy Iteration |
| Control | Bellman Optimality Equation | Value Iteration |

- **Iterative Policy Evaluation**：反复套用期望方程，评估 $v_\pi$。
- **Policy Iteration**：评估当前策略 → 对其贪心改进 (greedy policy improvement) → 再评估……交替直到收敛。
- **Value Iteration**：直接用最优方程，把 $\max$ 内嵌进每次更新，一步到位逼近 $v_*$，不显式维护中间策略。

> 一句话：期望方程是「评估」工具，最优方程是「优化」工具；Policy Iteration 把评估与改进拆开交替，Value Iteration 用 max 把两者揉进一次更新。

#### Policy Iteration vs Value Iteration

**Policy Iteration（两层循环）**

```
重复直到策略不再变化:
    ① 策略评估: 给定 π, 反复迭代 Bellman 期望方程, 直到 v_π 收敛   ← 内层要迭代很多步
    ② 策略改进: π'(s) = argmax_a [ R_s^a + γ Σ P·v_π(s') ]        ← 贪心改进一次
```

**Value Iteration（单层循环）**

```
重复直到 v 收敛:
    v(s) ← max_a [ R_s^a + γ Σ P·v(s') ]      ← 评估和改进揉进这一步
最后才提取策略: π*(s) = argmax_a [ R_s^a + γ Σ P·v*(s') ]
```

核心区别：Value Iteration 相当于「**只做一步评估就立刻改进**」的 Policy Iteration——它不等 $v_\pi$ 收敛，每扫一遍状态就用 $\max$ 顺带把改进做了。

| | Policy Iteration | Value Iteration |
|---|---|---|
| 评估 | 内层迭代到 $v_\pi$ **收敛** | 每个状态**只更新一次**（不等收敛） |
| 改进 | 评估完后单独做一次 argmax | $\max$ 直接内嵌在更新里，无独立改进步 |
| 显式策略 | 全程维护 $\pi$ | 中途不维护，**最后**才提取 $\pi^*$ |
| 每轮开销 | 大（内层多步） | 小（一次 backup） |

> 两者是同一思想的两端：中间还有 **truncated policy iteration**——评估只迭代 $k$ 步就改进。$k=\infty$ 是 Policy Iteration，$k=1$ 就是 Value Iteration。

#### Greedy Policy Improvement vs Bellman Optimality Equation

二者用的是**同一个 max 运算**，区别在于「输出什么」和「用谁的价值」：

| | Greedy Policy Improvement | Bellman Optimality Equation |
|---|---|---|
| 运算 | $\arg\max_a$ | $\max_a$ |
| 输出 | 一个**策略**（选哪个动作） | 一个**价值**（最好能值多少） |
| 用的价值 | 当前**固定策略**的 $v_\pi$ | 最优价值 $v_*$（或其当前估计） |

- **Greedy improvement** 拿着评估好的 $v_\pi$，问「每个状态改挑哪个动作最划算」，输出动作：

$$
\pi'(s) = \arg\max_{a}\left( \mathcal{R}_s^a + \gamma \sum_{s'} \mathcal{P}_{ss'}^a\, v_\pi(s') \right)
$$

- **Bellman 最优方程** 问「最优情况下这个状态值多少」，输出数值：

$$
v_*(s) = \max_{a}\left( \mathcal{R}_s^a + \gamma \sum_{s'} \mathcal{P}_{ss'}^a\, v_*(s') \right)
$$

关键联系：$\arg\max$（要动作）和 $\max$（要数值）只差一步——取了 $\arg\max$ 的动作所对应的那个值，正好就是 $\max$。所以 Value Iteration 里的一次 $\max$ backup，等价于「用当前 $v$ 做一次贪心改进 + 用改进后的策略评估一步」的合并。这就是为什么 Value Iteration 不需要单独的改进步——$\max$ 已经把贪心改进包含进去了。

> 一句话：greedy improvement 是 $\arg\max$（产策略），optimality equation 是 $\max$（产价值），二者是一枚硬币的两面。

#### Value Iteration 的单轮复杂度

对 $N$ 个状态、$m$ 个动作，一次完整迭代（扫一遍所有状态）的复杂度是 $O(mN^2)$。把更新公式拆成三层循环就清楚了：

$$
v(s) \leftarrow \max_{a \in \mathcal{A}} \Big( \mathcal{R}_s^a + \gamma \underbrace{\sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a\, v(s')}_{\text{遍历所有后继 }s'} \Big)
$$

| 层级 | 循环对象 | 次数 |
|---|---|---|
| 外层 | 更新每个状态 $s$ | $N$ |
| 中层 | 对每个动作 $a$ 算一遍（为取 max） | $m$ |
| 内层 | 求和需遍历所有后继 $s'$ | $N$ |

三层相乘即 $N \times m \times N = mN^2$。其中一个 $N^2$ 来自「每个状态都要看向所有其他状态」——转移 $\mathcal{P}_{ss'}^a$ 在最坏情况下稠密，每更新一个状态都要对 $N$ 个后继加权求和；再乘上 $m$ 是因为要在 $m$ 个动作里取 $\max$。等价地说，$\mathcal{P}$ 本身就是 $N \times m \times N$ 的张量，每轮迭代相当于把它整个过一遍。

**稀疏转移：$mN^2$ 只是上界。** 内层是 $N$ 还是常数，取决于每个 $(s,a)$ 的**后继分支数 (branching factor)**——即 $\mathcal{P}_{ss'}^a$ 这一行有多少非零项：

- **稠密 MDP**：可转移到任意状态 → 分支数 $\sim N$ → $mN^2$。
- **稀疏 MDP**（grid world、确定性环境、多数实际问题）：每个 $(s,a)$ 只能到达**常数个**后继 → 内层从 $N$ 降为 $O(1)$ → 退化到 $O(mN)$。

以 David Silver 课上的 grid world 为例：动作只有 $m=4$（上下左右），且环境**确定性**——选定动作后只会转移到**唯一**后继状态，$\mathcal{P}_{ss'}^a$ 那一行只有 1 个非零项。于是内层求和不必遍历 $N$ 个状态，只看那 1 个后继：

| 层级 | 一般情形 | Grid world |
|---|---|---|
| 外层（状态 $s$） | $N$ | $N$ |
| 中层（动作 $a$） | $m$ | $m=4$ |
| 内层（后继求和） | $N$ | $O(1)$（确定性，只到 1 个后继） |
| **单轮总计** | $mN^2$ | $O(mN) = O(N)$ |

即使是随机版本（如「80% 按预期方向、各 10% 滑向两侧」），分支数也只有 3，仍是常数，依旧 $O(mN)$。所以 grid 例子能手算、能快速收敛，正是因为它落在 $O(mN)$ 这一端。

#### Synchronous vs Asynchronous Backup

上面算的单轮复杂度（$mN^2$ 或稀疏的 $mN$）有一个隐含前提：**synchronous backup（同步备份）**，即一次 sweep 要扫完全部 $N$ 个状态。

**Synchronous** 用上一轮的整张价值表 $v_k$ 算出新的整张表 $v_{k+1}$，所有状态「同时」更新：

$$
v_{k+1}(s) \leftarrow \max_a \Big( \mathcal{R}_s^a + \gamma \sum_{s'} \mathcal{P}_{ss'}^a\, v_k(s') \Big), \quad \forall s \in \mathcal{S}
$$

- 每轮**必须扫全部 $N$ 个状态**——这正是单轮复杂度 $mN^2$ 的来源。
- 需要**两份表**：读 $v_k$、写 $v_{k+1}$，整轮算完再替换；本轮内一个状态的更新不会影响其他状态（都用同一份旧值）。

**Asynchronous backup（异步备份）** 打破「必须扫全部、且用同一份旧值」这两个前提：

| | Synchronous | Asynchronous |
|---|---|---|
| 每轮更新范围 | 全部 $N$ 个状态 | 任意子集 / 单个状态 |
| 用的价值 | 统一的旧表 $v_k$ | **就地 (in-place)** 更新，可立刻用上刚算出的新值 |
| 存储 | 两份表 | 一份表即可 |

几种典型异步法：

- **In-place DP**：只存一份表，更新 $s$ 时直接覆盖，后续状态立刻用上新值，往往收敛更快。
- **Prioritized sweeping**：按 Bellman error 排优先级，优先更新「变化最大」的状态。
- **Real-time DP**：只更新智能体实际访问到的状态。

只要每个状态被**持续选中**更新，异步备份仍能收敛到 $v_*$，但能把算力集中在重要状态上，避免每轮机械地死扫全部 $N$ 个。

> 注：以上为 David Silver 强化学习课程中的经典表述。
