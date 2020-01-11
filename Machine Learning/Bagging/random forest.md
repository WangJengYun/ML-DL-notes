---
tags: Machine Learning,Bagging
---
Random Forest
===

## From Bootstrap to Bagging
Bootstrap是一種非參數的統計方法，運用在各個領域，主要針對觀測訊息，進而對總體的分布特性進行統計推斷，具有穩健性(robust)，其中**Bootstrap Variance Estimator**定義如下:

Let $X_1,X_2,...,X_n∼P$ , $P$ is the empirical distribution and an $iid$ of size sample drawn from $P_n$ is a bootstrap samlpe, denoted by$$
X_1^*,X^*_2,...,X^*_n∼P\tag{1}
$$Now suppose we use the bootstrap algorithms for estimating the variance of $\hat \theta$,let $\hat \theta=g(X_1,X_2,...,X_n)$ denote some estimator.We would like to find the variance of $\hat \theta$ as following:$$
var_P(\hat \theta)=var(g(X_1,X_2,...,X_n))≡s(P)\tag{2}
$$
1. Draw $X^*_1,X^*_2,...,X^*_n$ from $P$,where each bootstrap sample $X^*_b$ is a random sample of $n$ data points drawn $iid$ from P with replacement.
2. Calculate $\hat \theta^*_b$ = ,where $\hat \theta^*_b= g(X^{*(b)}_1,X^{*(b)}_2,...,X^{*(b)}_n)$; there are called the bootstrap replications
3. Let $s^2$ be the sample variance of $\hat \theta^{*}_1,\hat \theta^{*}_2,...,\hat \theta^{*}_B$,so$$
s^2=\frac{1}{B}\sum^B_{b=1}(\hat \theta_j^*-\bar \theta)^2=\frac{1}{B}\sum^{B}_{b=1}(\hat \theta_j^*)^2-(\frac{1}{B}\sum^{B}_{b=1}\hat \theta_j^*)^2\tag{3}
$$

使用Bootstrap方法進行抽樣，生成多個集合的樣本推估母體參數，並可以藉由平均多個集合而降低參數估計的變異數，所以減少變異的概念會拿來運用至機器學習，則會提升模型準確性，數學表示如下:

Let traning set is $D = \{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\}$, where each $x_i$ belongs to some domain or instance space $X$, and each $y_i$ is in some label set $Y$

1. Draw $D^*_1,D^*_2,...,D^*_B$ from $D$, where each bootstrap sample $D^*_b=\{(x^{*b}_1,y^{*b}_1),(x^{*b}_2,y^{*b}_2),(x^{*b}_m,y^{*b}_m)\}$ is drawn from $D$ with replacement.
2. Obtaining the predict model $\hat f(x^*)$ at input $x^*$
3. averaging all predict result, is defined by $$
\hat f_{avg}(x*)= \frac{1}{B}\sum^B_{b=1}\hat f^b(x^*)\tag{4}
$$

因次在機器學習這裡會對訓練集(training set)進行使用Bootstrap方法進行抽樣，假設抽出B組訓練集，並分別建立預測模型，最後將模型結果進行平均得到最終預測結果，這個方法我們稱為Bagging。


