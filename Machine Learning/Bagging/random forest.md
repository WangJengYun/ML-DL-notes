---
tags: Machine Learning,Bagging
---
Random Forest
===
隨機森林是目前最常使用的機器學習演算法之一，主要是基於決策樹(decision tree)的組合驗算法，由(Breiman 2001)提出。原始的隨機森林的演算法是由CART(Classification and regression tree)所建構的，並透過Bagging演算法建置相互無相關的樹，最後再平均每棵樹的預測結果。

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

使用Bootstrap方法進行抽樣，生成多個集合的樣本推估母體參數，並可以藉由平均多個集合而降低參數估計的變異數，所以減少變異的概念拿來運用至機器學習，則會提升模型準確性，數學表示如下:

Let traning set is $D = \{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\}$, where each $x_i$ belongs to some domain or instance space $X$, and each $y_i$ is in some label set $Y$

1. Draw $D^*_1,D^*_2,...,D^*_B$ from $D$, where each bootstrap sample $D^*_b=\{(x^{*b}_1,y^{*b}_1),(x^{*b}_2,y^{*b}_2),(x^{*b}_m,y^{*b}_m)\}$ is drawn from $D$ with replacement.
2. Obtaining the predict model $\hat f(x^*)$ at input $x^*$
3. averaging all predict result, is defined by $$
\hat f_{avg}(x*)= \frac{1}{B}\sum^B_{b=1}\hat f^b(x^*)\tag{4}
$$

因次在機器學習這裡會對訓練集(training set)進行使用Bootstrap方法進行抽樣，假設抽出B組訓練集，並分別建立預測模型，最後將模型結果進行平均得到最終預測結果，這個方法我們稱為Bagging。
## Random Forest Algorithm
了解Bagging的概念之後，知道Bagging似乎在高變異與低偏差的過程是可以有不錯的效果，此過程類似決策樹(decision tree)，接我們開始建構隨機森林模型，如下:

Assume we observe a dataset of $D={(x_i,y_i)}_{i=1,...,n}$ ,where $x_i = (x^1_i,x^2_i,...,x^m_i)$ is a random vector of $m$ features and $y_i \in \mathbb{R}$ is a target,which can be either binary or numerical response.<br>
1. For b=1 to B:<br>
    1. Draw a bootstrap sample $D_t^*$ from $D$ by sampling $|D|$ data points with replacement.<br>
    2. Grow a random-forest tree $T_b$ to the boostsrapped data, by recursively repeating the following steps for each terminal node of the tree,until the mininum leaf node size and maximum tree depth is reached.
        * Select k festures at random from the m features,where k<<m
        * Pick the best feature/split-point among the k
        * split the node into two daughter nodes
2. Output the ensemble of trees $\{T_b\}^B_i$<br>
    To make a prediction at a new points $x$:<br>
    * Regression : $\hat{f^B}(x) = \frac{1}{n}\sum^B_{b=1} T_b(x)$
    * Classification : Let $\hat{C}_b$ be the class prediction of the $b$th random forest tree.Then $\hat{C}^B = majority\,\,vote \{\hat{C}_b(x)\}^B_1$<br>

上述的演算過程，是利用Bagging方法，取後放回的方式抽樣，這樣可以使random forest變異降低，另外藉由這技巧，每顆樹是不需要進行剪枝，但每顆樹所需要特徵事先要隨機抽樣，因為我們希望每顆樹彼此之間是無相關性，最後$B$顆樹進行聚合時，如果任務為分類，則簡單多數投票法，反之為迴規則用平均法，得到我們最終預測結果。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/random%20Forest/structure.PNG?raw=true)
圖片來源(https://rstudio-pubs-static.s3.amazonaws.com/378052_30d987a09ea54b6db5aa1e82f5dce6bf.html)
## Variance of random Forest 
本章節來探討random forest的tree變異度，變異程度是會影像模型的準確度，將每顆樹視為一個隨機變數台探討:<br>
From (),we redefine the model.
After B such trees $\{T(x;\theta_b)\}^B_i$ are grown,the random forest (regressor) predictor is
$$\hat{f}(x)=\frac{1}{B}\sum^B_{b=1}T(x,\theta_b)$$
where $\theta_b$ characterizes the $b$th random forest tree in term of split feature cutpoints at each node,and terminal-node values.Assume $\{T(x;\theta_b)\}^B_i$ is i.d variables (identically distributed,but not neccessarily independent) 

* $\sigma^2$(x) is the sampling variance of any single randomly drawn tree. 
$$ var(T(x;\theta_b)) = \sigma^2(x)$$
* $\rho(x)$ is the sampling correlation between any pair of random forest used in the average:
$$\rho(x) = var(T(x,\theta_{b_i}),T(x,\theta_{b_j})),where 1\leq i,j\leq B$$
where $\theta_{b_i},\theta_{b_j}$ are a randomly drawn pair of random forest tree grown.<br>

Now the variance of random forest : 
$$ var(\hat{f}(x))=\rho(x)\sigma^2(x)+\frac{1-\rho(x)}{B}\sigma^2$$
if $B\rightarrow \infty$,$var(\hat{f}(x))=\rho(x)\sigma^2$

由上是表示說當我們建構的樹越多，則再X的第二項則會趨近於0，但不能表示說樹越多則變異越少，因為隨機森林的變異還跟樹之間的相關性有關，如果相關性越少則會減少變異，所以說在建構樹時，盡量不要太過於相似，則模型的表現交差，因次我們在建構樹時，會再隨機挑選特徵，使樹之間會有差異性，然而提出隨機森林的作者提供建議值如下:<br>
* 對於分類任務，k的默認值為$\sqrt{m}$與每個葉節點的樣本樹最小值1。
* 對於迴歸任務，k的默認值為$\frac{1}{m}$與每個葉節點的樣本樹最小值5。
## Out of Bag Samples
Out of Bag 是random forest特有性質，主要是random forest驗證方式，類似於交叉驗證(cross validation)，交叉驗證主要會將資料集做切分一部分拿來做訓練，一部分拿來做驗證，但OOB是不用事先做資料切割，而是利用bootstrap性質，將沒有被抽到的樣本視為out of bag，再利用這些計算score，數學公式如下:
$$\hat{Err}^{OOB}(f) = \frac{1}{n}\sum_{(x_i,y_i) \in D} L(\frac{1}{B^{-i}}\sum_{b=1}^{B^{-i}}\hat{f_{m_b}}(x_i),y_i)$$
where $m_1,m_2,...,m_{B^{-1}}$ denote the indices of $B^{-i}$ the trees that have been built from bootstrap replicate that do not include $(x_i,y_i)$ and L(.) is the loss/error function.

由上述得知，我們再計算每筆樣本的OOB，是將沒有將這筆拿來訓練的tree的結過進行平均，所以每一筆樣本所計算OOB的樹是會不相同的，反之每顆樹拿來做OOB驗證也是不同樣本。另外整個資料中，其實會有近$\frac{1}{3}$的資料不會被抽到，證明如下:

We suppose that there are n samples in the training dataset,then the probability of not picking a sample in a random draw is $$\frac{n-1}{n}$$The probability of not picking n samples in random draws is $$(\frac{n-1}{n})^n$$
When $n \rightarrow \infty$
$$lim_{n \rightarrow \infty}(1-\frac{1}{n})^n = e^{-1}=0.368$$
因此對於每顆樹大約36.8%的訓練集視為OOB樣本。OOB的機制是會比交叉驗證來的省資源，當你資料量不多時，OOB會是不錯的選擇，因為每個樣本在不同tree可當訓練集或者驗證集，是節省資源採樣的估計方法。
## feature important 
最後我們介紹特徵重要性，其實這個性質是來自於決策數(decision tree)，再決策樹我們會利用純度指標來衡量每個節點為同一類的程度，最常被使用的為entropy與gini index，其公式如下:<br>
* $Entropy(s_t) = \sum_{i=1}^C -p_ilog(p_i)$
* $Gini(s_t) = 1-\sum_{i=1}^C p_i^2$<br>

where $p_i$ is the probability of samples belonging to class $i$ 

然而我們藉由上述的純度測量為基礎，來衡量每一個切點或特徵的重要程度，就是切割資料之後，其純度的減少量可當切點或特徵的重要程度如下:

we identifies at each node $t$ the split $s_t$ for which the partition of $n_t$ node samples into $t_L$ and $t_R$ maximizes the decrease 
$$\nabla i(s,t) = i(t)-p_Li(t_L)-p_Ri(t_R)$$
where $p_L=\frac{n_{t_L}}{n_t}$ and $p_R=\frac{n_{t_R}}{n_t}$

上述為決策數來衡量變數的重要程度，然而random forest採用這個姓值計算整體的重要程度，是以不同顆樹進行平均，如下:

we evaluate the importance of $x_m$ for predict $y$ by adding up the weighted impurity decreases $p(t)\nabla i(s_t,t)$ for all nodes t where $X_m$ is used,averaged over all B trees in the forest:
$$Imp(x_m) = \frac{1}{B}\sum_{b=1}^B\sum_{t \in b:v(s_t)=X_m}p(t)\nabla i(s,t)$$
where p(t) is the proportion $\frac{n_t}{n}$ of samples reaching t and $v(s_t)$ is the feature used in split $s_t$
上述視為主要衡量特徵重要性，如果我們採用entropy來測量純度，則上述公式稱為 Mean Decrease Impurity importance (MDI)，反之我們使用 Gini index，則稱為Gini importance或 Mean Decrease Gini。

