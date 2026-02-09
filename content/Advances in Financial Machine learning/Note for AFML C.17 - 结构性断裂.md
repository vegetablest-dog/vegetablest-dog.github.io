---
title: Note for AFML C.17 - 结构突破
tags:
  - feature
---
## 结构性断裂
在开发基于机器学习的投资策略时，我们通常希望在存在多个因素汇聚且其预测结果提供有利的风险调整回报时进行投资。结构性断裂，比如市场从一种状态过渡到另一种状态，就是此类特别有吸引力的汇聚因素的一个例子。
### 结构断裂测试类型
可以将结构性断点检验分为两类：
- CUSUM 检验：这些检验用于判断累计预测误差是否显著偏离白噪声。
- 爆炸性测试：除了偏离白噪声之外，这些测试还用于判断过程是否表现出指数增长或崩溃，因为这与随机游走或平稳过程不一致，而且从长远来看是不可持续的。
	- 右尾单位根检验：这些检验在假设自回归模型的基础上，用于评估是否存在指数增长或崩溃。
	- 下/上鞅检验：这些检验评估在各种函数形式下是否存在指数增长或崩溃。
### CUSUM检验
我们之前介绍了CUSUM滤波器[[Note for AFML C.2 - 数据结构#信息驱动的柱]]，并将其应用于基于事件的K线采样中。其思路是在某个变量（如累积预测误差）超过预设阈值时对K线进行采样。这个概念可以进一步扩展，用于检测结构性突变。
#### 递归残差的Brown-Durbin-Evans CUSUM检验
假设在每个观测点$t=1,...,T$，使用特征$x_t$预测目标值 $y_t$. 使用递归最小二乘估计
$$y_t = \beta_t^\prime x_t + \varepsilon_t$$
在子样本集$([1,k+1],[1,k+2],...,[1,T])$上得到$T-k$个系数估计$(\hat{\beta}_{k+1}, ..., \hat{\beta}_T)$. 进一步计算标准化的一步前向递归残差
$$\hat{\omega}_t = \frac{y_t - \hat{\beta}_{t-1}^\prime x_t}{\sqrt{f_t}},$$
where
$$f_t = \hat{\sigma}_{\varepsilon}^2[1+ x_t^\prime(X_t^\prime X_t)^{-1}x_t].$$
(这个$\hat{\sigma}_{\varepsilon}^2$是怎么计算的？是直接拟合残差的方差吗)
CUSUM统计量定义为
$$S_t = \sum_{j=k+1}^t \frac{\hat{\omega}_j}{\hat{\sigma}_\omega},$$
where
$$\hat{\sigma}_{\omega}^2 = \frac{1}{T-k} \sum_{t=k+1}^T (\hat{\omega}_t - E[\hat{\omega}_t])^2.$$
在零假设下，$\beta$是常数，则$S_t \sim \mathcal{N}(0,T-k-1)$. 这个过程的一个注意事项是起点是任意选择的，因此结果可能会不一致。
#### Chu-Stinchcombe-White 水平CUSUM检验
它丢弃了$\{x_t\}$并假设$H_0:\beta_t =0$, 即$E_{t-1}[\Delta y_t]=0$. 计算对数价格$y_t$相对于对数价格$y_{n}, t>n$的标准化偏差
$$S_{n,t} = (y_t - y_n)(\hat{\sigma}_t\sqrt{t-n})^{-1},$$
where
$$\hat{\sigma}_t^2 = (t-1)^{-1}\sum_{i=2}^t (\Delta y_i)^2.$$
在零假设$H_0:\beta_t =0$下，$S_{n,t} \sim \mathcal{N}(0,1)$. 单侧检验的时间依赖临界值为$c_{\alpha}(n,t) = \sqrt{b_{\alpha + \log(t-n)}}.$
参考值$b_{0.05} = 4.6$. 该方法的一个缺点是参考水平$y_n$设置得有些随意。为了克服这一缺陷，我们可以在一系列向后移动的窗口$n \in [1,t]$上估计$S_{n,t}$ 然后选择$S_t = \sup_{n\in [1,t]} S_{n,t}$.
### 爆炸性测试
爆炸性测试通常可以分为两类：一类测试单个泡沫，另一类测试多个泡沫。允许多个泡沫的测试在某种意义上更为稳健，因为泡沫-破裂-泡沫的循环会使序列在单泡沫测试中看起来是平稳的。
#### Chow-Type Dickey-Fuller检验
考虑一阶自回归过程：
$$y_t = \rho y_{t-1} + \varepsilon_t,$$
where $\varepsilon$ is white noise.  零假设是接着$y_t$随机游走，$H_0:\rho=1$, 但在时间$\tau^* T$处发生突变，$\tau^* \in (0,1)$. 变为如下形式
$$H_1: y_t = y_{t-1} + \varepsilon_t, \text{  for } t\leq\tau^*T; y_t = \rho y_{t-1} + \varepsilon_t,\text{  for } t>\tau^*T \text{  with } \rho >1.$$
为了检验断点，我们拟合
$$\Delta y_t = \delta y_{t-1} D_t[\tau^*] + \varepsilon_t,$$
其中$D_t[\tau^*]$是一个哑变量，当$t< \tau^* T$时为0，当$t\ge \tau^*T$时为1. 零假设为$H_0:\delta = 0$对比备择假设$H_1:\delta >1$计算检验统计量：
$$DFC_{\tau^*} = \frac{\hat{\delta}}{\hat{\sigma}_\delta}.$$
这一方法的主要缺点是$\tau^*$是未知的。可以在区间$[\tau_0, 1-\tau_0]$中尝试所有可能的$\tau^*$.但是我们需要保证，两种状态都有足够的观测值用于拟合。则最终的检验统计量为$SDFC = \sup_\tau DFC_{\tau}$.
另一个主要缺点是这个方法只假设了一个断点，对于多个状态，我们需要新的检验方法。
#### 极大增强Dickey-Fuler检验(SADF)
*标准单位根和协整检验并不适合作为检测泡沫行为的工具，因为它们无法有效区分平稳过程和周期性崩溃的泡沫模型。数据中周期性崩溃泡沫的模式更像是由单位根或平稳自回归生成的数据，而非潜在的爆炸性过程。*
为了解决这一问题，可以拟合模型
$$\Delta y_t = \alpha + \beta y_{t-1} + \sum_{l=1}^L \gamma_l \Delta y_{t-l} + \varepsilon,$$
其中，假设为$H_0:\beta \leq 0, H_1:\beta>0$. SADF在每个端点$t$使用向后拓展的起点拟合上述回归，然后计算：
$$SADF_t = \sup_{t_0\in [1,t-\tau] } ADF_{t_0,t}= \sup_{t_0\in [1,t-\tau] } \frac{\hat{\beta}_{t_0,t}}{\hat{\sigma}_{\beta_{t_0,t}}}$$
其中$\hat{\beta}_{t_0,t}$是从$t_0$开始到$t$结束的样本上估计的，$\tau$是分析中使用的最小样本长度，$t_0$是向后拓展窗口的左边界，$t = \tau,...,T$.  为了估计$SADF_t$, 窗口的有边界固定在$t$. 标准的$ADF$测试是$SADF_t$在$\tau = t-1$时的特例。
$SADF_t$和$SDFC$之间有两个关键区别，$SADF_t$在每个$t\in [\tau,T]$处计算，而$SDFC$只在$T$处计算。其次，$SADF$是递归地展开样本地起始部分。通过尝试$(y_0,t)$地双重循环地所有组合，$SADF$可以步假定断点次数和时间。
##### 原始价格 vs. 对数价格
在文献中，常见的做法是对原始价格进行结构性断裂检验。在本节中，我们将探讨为什么应优先使用对数价格，尤其是在处理涉及泡沫和崩溃的长时间序列时。
对于原始价格序列，ADF地零假设被拒绝意味着序列是平稳的，具有有限方差。其含义是，在波动时，价格上下运行时的差是恒定的，而不是收益率的方差是稳定的。如果收益方差恰好对价格水平不变，该模型将具有结构性异方差。
相对的，如果使用对数价格，可以更加关注收益率的稳定。在小样本中，这种差异在实际中可能无关紧要，因为 $k \approx 1$ ，但 SADF 跨数十年运行回归，且泡沫会导致各阶段的价格水平显著不同$(k \ne 1)$。
##### 计算复杂度
SADF的算法复杂度是$O(n^2)$， 有$\sum_{t= \tau}^T t-\tau+1$次ADF检验。考虑ADF的矩阵表示，每个ADF回归需要$O(N^3) + O(N^2T)$的浮点运算。这个计算是很复杂的。
##### 指数行为的条件
考虑零滞后的对数价格模型，$\Delta\log y_t = \alpha + \beta \log y_{t-1} + \varepsilon_t.$ 整理可得$\log \tilde{y}_t = (1+\beta)\log \tilde{y}_{t-1} + \varepsilon_t,$ 其其中$\log \tilde{y}_t = \log y_t + \alpha/\beta$.  滚动$t$步，得到$E[\log \tilde{y}_t] = (1+\beta)^t \log \tilde{y}_0$. 这表明：
- 稳定性：$\beta <0 \Rightarrow \lim_{t \to \infty} E[\log y_t] = -\alpha/\beta.$ 半衰期为$t = -\frac{\log 2}{\log(1+\beta)}.$
- 单位根：$\beta = 0$, 这时系统不稳定，表现为鞅
- 爆炸增长：$\beta > 0$.
##### 分位数ADF
SADF是取$t$值序列的上确界. 选择极值会引入稳健性问题。可以使用分位数估计法：
1. $s_t = \{ADF_{t_0,t}\}_{t_0 \in [0, t_1-\tau]}.$
2. 定义$Q_{t,q}$是$s_t$的$q$分位数, $q \in [0,1]$.
3. 定义$\dot{Q}_{t,q,v} = Q_{t,q+v} - Q_{t,q-v}$, $0 < v \leq \min\{q,1-q\}$, 作为高ADF值得离散程度测度。例如，$q=0.95, v=0.025$.
##### 条件ADF
也可以通过计算条件矩来解决SADF稳健性的问题。设$f(x)$为$s_t$的概率分布函数。定义$C_{t,q} = K^{-1} \int_{Q_{t,q}}^\infty xf(x)dx$来衡量高ADF值的中心值， $\dot{C}_{t,q} = \sqrt{K^{-1}\int_{Q_{t,q}}^\infty (x- C_{t,q})^2 f(x) dx}$ 来衡量高ADF值的离散程度，正则化常数$\int_{Q_{t,q}}^\infty f(x)dx.$ 可以使用$q = 0.95$.
##### SADF的实现
详见书中，略。
#### 次鞅和超鞅检验
考虑一个过程，是次鞅或者超鞅。观测值$\{y_t\}$, 检验爆炸性时间趋势存在, $H_0: \beta=0, H_1:\beta\ne 0$, 在以下不同的备择条件下：
- 多项式趋势：$y_t = \alpha + \gamma t + \beta t^2 + \varepsilon_t.$
- 多项式趋势：$\log y_t = \alpha + \gamma t + \beta t^2 + \varepsilon_t.$
- 指数趋势：$y_t = \alpha e^{\beta t} + \varepsilon_t$.
- 幂指数趋势：$y_t = \alpha t^\beta + \varepsilon_t$.
类似SADF，对这些假设进行拟合，并计算
$$SMT_t = \sup_{t_0 \in [1,t-\tau]} \frac{|\hat{\beta}_{t_0,t}|}{\hat{\sigma}_{\beta_{t_0,t}}}.$$
使用绝对值的原因是因为我们对爆炸性增长和崩溃同样感兴趣。
弱长期泡沫的$\hat{\sigma}_{\beta}^2$可能小于强短期泡沫的$\hat{\sigma}_{\beta}^2$，因此这种方法容易偏向长期泡沫，可以稍微加入调整项$\varphi \in [0,1]$:
$$SMT_t = \sup_{t_0 \in [1,t-\tau]} \frac{|\hat{\beta}_{t_0,t}|}{\hat{\sigma}_{\beta_{t_0,t}}(t-t_0)^\varphi}.$$
当$\varphi \to 0$, $SMT_t$将表示处更长期的趋势；当$\varphi \to 1$, $SMT_t$将变得嘈杂，因为选入了更多的短期泡沫。
以便它能够筛选针对特定持有期的机会。机器学习算法使用的特征可能包括从更广范围的𝜑值估计的SMT。