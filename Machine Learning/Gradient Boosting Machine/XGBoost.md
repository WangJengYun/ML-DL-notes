---
tags: Machine Learning
---
XGBoost
===
XGB(Extreme Gradient Boosting)基於gradient boosting tree所提出高效能的機器學習演算法之一，在2016年陳天奇博士所提出的，目前也是被常用於Kaggle比賽中，此演算法相對於傳統的gradient boosting tree來說，同時考慮系統優化及數學理論方法，用最少的執行時間與效能能夠提高模型準確性。
## 正規化目標函式(Regularized objective function)
傳統的gradient boosting tree主要以決策樹(decision tree)，當作弱學習器 (weak learner)，最後將所有的弱學習器進進行加總，得到最後的預測結果，如下圖:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/XGBoost/twocart.png?raw=true)

但決策樹(decision tree)有個缺點為較容易overfitting，然而XGB解決這個問題，在目標函式加入了正規項/懲罰項，限制樹的生長，進而避免overfitting，定義如下:

Suppose that we have a dataset with n examples and m feature $D=\{(x_i,y_i)\}$ , where $x_i \in \mathbb{R}^m$,$y_i \in \mathbb{R}$. A tree ensemble model use K additive functions to predict the output:
$$\hat{y}_i = \phi(x_i) = \sum^K_{k=1}f_k(x_i),\,\,f_k \in F$$
where 
* $F = \{f(x) = w_{q(x)}\}(q:\mathbb{R}^m\rightarrow T,w\in\mathbb{R}^T)$ is the space of regression trees(also known as CART).
    * $q$ represents the structure of each tree that maps an exmaple to the corresponding leaf index.
    * $T$ is the number of leaves in the tree
    * Each $f_k$ corresponds to an independent tree structure $q$ and leaf weights $w$.<br>

Unlike decision tree, each regression tree contains a continuous scorce on each of the leaf,we use $w_i$ to represent scores on $i$th leaf.For a given example, we will use decision rules in the tree (given by $q$) to classify it to the leaves and calculate the final prediction by summing up the score in the corresponding leaves(given by $w$).

To learn the set of function used in the model, we minimize the following regularized objection:
$$L(\phi) = \sum_{i=1}^nl(\hat(y_i),y_i)+\sum_{k=1}^K \Omega(f_k),\,\,\,\,where\,\ \Omega(f) = \gamma T+\frac{1}{2} \lambda||w||^2$$
Here 
* $l$ is a differentiable convex loss funciton that measures the difference between the predict    $\hat{y}_i$ and the target $y_i$.
* The second $\Omega$ penalizes the complexity of the model(i.e.,the regression tree funciton)

由定義正規化目標變數，可知道在目標變數上加上正規項/懲罰項可避免overfitting，正規項是由兩個表示項來表示模型的複雜度，可分為葉子節點的數量$(T)$與葉子節點的權重平方和$(||w||^2)$。然而當$\gamma$和$\lambda$為0時，則目標函數就會是原來傳統gradient boosting tree。

### Dark

### Shrinkage and Column Subsampling
除了利用正規化方式避免overfitting，在XGB演算法中，也可使用Shrinkage與特徵抽樣(column subsampleing)，祥細說明如下:
* **Shrinkage** : 在每次迭代後的葉子節點分數添加縮減權重，希望要減少每一顆樹影響，保留一些空間給未來的樹去優化模型
* **特徵抽樣(column subsampleing)** : 類似於隨機森林(random forest)，針對特徵空間進行抽樣，然後使用所選取的特徵建立弱學習器，另外作者提到特徵抽樣是會比傳統的樣本抽樣(row sub-sampling)更能預防overfitting。

## Gradient Boosting tree of XGBoost
接著我們來說明XBG的Gradient Boosting，由前一張我們了解Gradient Boosting的基本概念，主要是透過新的弱學習器，來訓練前面所有的弱學習器的殘差，且透過梯度下降法(gradient  descent)，來找到最佳的弱學習器，來最小化損失函式，其數學原理如下:

Let $\hat{y}^{t}_i$ be the prediction of the $i$th instance at $t$th iteration, we will need to add $f_t$ to minimize the following objective.

$$
\begin{aligned}
L^t &= \sum^n_{i=1}l(y_i,\hat{y}^{t-1}_i+f_t(x_i))+\Omega(f_t)\\
&\approx \sum^n_{i=1}[l(y_i,\hat{y}_i)+g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)\,\,\,\,by\,\,Taylor \,\,expression\\
&=\sum^n_{i=1}[g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)\,\,\,\,removing\,\,the \,\,constant\,\,terms\\
\end{aligned}
$$
where $g_i=\frac{\partial l(y_i,\hat{y}_i^{t-1})}{\partial \hat{y}_i^{t-1}}$ and $h_i=\frac{\partial^2 l(y_i,\hat{y}_i^{t-1})}{\partial^2 \hat{y}_i^{t-1}}$ 
Define $I_j = \{i|q(x_i)=j\}$ as the instance set of leaf $j$,we can rewrite it by expanding the regulariztion term $\Omega$ as follows:
$$
\begin{aligned}
L^t &= \sum^n_{i=1}[g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i)]+\gamma T+\frac{1}{2}\lambda\sum^T_{j=1}w^2_j\\
&=\sum^T_{j=1}[(\sum_{i \in I_j}g_i)w_j+\frac{1}{2}(\sum_{i \in I_j}h_i+\lambda)w_j^2]+\gamma T
\end{aligned}
$$
For a fixed structure $q(x)$, we can compute the optimal weight $w^*_j$ of leaf $j$ by 
$$w_j^* = -\frac{\sum_{i \in I_j}g_i}{\sum_{i \in I_j} h_i+\lambda }$$
and calculate the corresponding optimal objective function value by 
$$L^t = -\frac{1}{2}\sum^T_{j=1}\frac{(\sum_{i \in I_j}g_i)^2}{\sum_{i \in I_j}h_i+\lambda}+\gamma T$$
it can be used as a scoring function to measure the quality of a tree structure $q$.

由上述的數學原理得知，我們藉由損失函數的一階梯度值與二階梯度值，來建構弱學習器，另外我們也可以使用Eq(X)來表示所建立的樹的品質，在計算樹的品質時，找到最佳的弱學習器，然而這可以讓我們可以客製化可微分損失函數，這個會是相當便利的，另外在計算過程中，我們只需要將每個個樣本的一階梯度值與二階梯度值進行加總，並考慮模型複雜度，得到品質分數，如下圖:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/XGBoost/struct_score.png?raw=true)
最後XGB的整個演算過程如下:
(1) $f_t=arg\,\,min_{\rho}\sum_{i=1}^nl(y_i,\rho)$
(2) For $t=1$ to $T$ do : 
(3) $\quad\quad g_i=\frac{\partial l(y_i,\hat{y}_i^{t-1})}{\partial \hat{y}_i^{t-1}},h_i=\frac{\partial^2 l(y_i,\hat{y}_i^{t-1})}{\partial^2 \hat{y}_i^{t-1}}$
(4) $\quad\quad f_t = arg\,\,min_{f_t} \sum_{i=1}^n\frac{1}{2}h_i((-\frac{g_i}{h_i})-f_t(x_i))^2+\Omega(f_t)+constant$
(5) $\quad\quad \phi_t(x) = \phi_{t-1}(x)+\eta_tf_t(x)$
(6) endFor
(7) end Algorithm
## Split Finding Algorithms
接著我們介紹XGB如何找到最佳切點。在決策樹(Decision tree)中，主要是以entropy與gini來表示每個節點的純度，然而XGB是採用一階梯度值與二階梯度值來計算純度如之前Eq()，並計算切割前原始的節點的純度與切割後的左右量的節點的純度的相減，稱為純度下降量，最後我們以純度的下降量來選擇切點，如果純度下降量越高表示此切點是能提高模型的準確度，反之較低則是無意義的切割，其數學原理如下:
Assume that $I_L$ and $I_R$ are the instance sets of left and right nodes after the split. Letting $I = I_L \cup I_R$,then the loss reduction after split is given by 
$$L_{split}=\frac{1}{2}[\frac{(\sum_{i \in I_L}g_i)^2}{\sum_{i \in I_L}h_i+\lambda}+\frac{(\sum_{i \in I_R}g_i)^2}{\sum_{i \in I_R}h_i+\lambda} - \frac{(\sum_{i \in I}g_i)^2}{\sum_{i \in I}h_i+\lambda}]-\gamma $$
根據上述的公式得知，如果 $L_{splite}$ 小於 $\gamma$，表示此樹的架構已經表現得相當不錯，不需要再切割資料，這就是決策樹(decision tree)的事後砍樹手法(pruning techniques)。

除了用純度下降量來估計每個切點的優劣之外，如何挑選切點的集合也是相對重要，當資料量較大時，要遍歷所有可能的切點計算純度下降量，這是會消耗很多時間，接著我們來介紹XGB如何使演算法更有效率地找到最優切割點。
### 貪婪演算法(Basic Exact Greedy Algorithm)
通常我們找到最佳切割點，主要的作法是會在所有的特徵中，計算將所有的可能的切割點，我們稱為貪婪演算法(exact greedy algorithm)，如下:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/XGBoost/Exact%20Greedy%20Algorithm%20for%20Split%20Finding.PNG?raw=true)
由上述的演算法，可知為了更有效率找到最佳的切割點，會先將樣本依照特徵值進行排序，遍歷所有的可能的切割點，加總其梯度值，然後再利用Eq，尋找最佳切割點，如下圖:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/XGBoost/split_find.png?raw=true)
### 近似演算法(Approximate Algorithm)
貪婪演算法雖然可以找到最優解，但是當資料量太大則無法讀到記憶體進行計算，會有這樣的缺點，所以作者提出近似演算法。近似演算法會先依據特徵分布的分位數提出候選切割點，然後將連續型特徵根據候選切割點進行分組，計算每組的統計資訊，並在這些候選切割點找到最佳的切割點，然而提出候選切割點時會有兩個策略:
* Global : 在學習每一顆樹之前就提出候選切割點，並在每次切割都採用這些候選切割點。
* Local : 每次切割前會重新得出候選切割點。

演算過程如下
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/XGBoost/Approximate%20Algorithm%20for%20Split%20Finding.PNG?raw=true)

值觀上來看，Global策略會是比local策略需要更多的候選切割點，因為local策略在每次切割前會重新提出候選切割點，這樣是會考慮到切割前的節點特徵分布情況，這會是比較適合成長較深的樹，但是這樣計算量就會變多。作者有根據這兩個情況進行實驗如下圖，會發現Global在畫分點較多時(eps較小)與local(eps大)有差不多的精確度，另外可以發現，在給予合理的eps值，近似演算法會是與貪婪演算法有相同的精確度。
![](https://raw.githubusercontent.com/WangJengYun/ML-DL-notes/master/Machine%20Learning/images/XGBoost/test.PNG)
#### Weighted Quantile Sketch
在近似演算法中，最重要的一個步驟為如何提出候選切割點，作者提出Weighted Quantile Sketch來定義候選切割點，如下:
Let multi-set $D_k = \{(x_{1k},h_1),(x_{2k},h_2),...,(x_{1n},h_n)\}$ represent the $k$th feature values and seconde order gradient statistics of each training instances.We can define a rank function $r_k:\mathbb{R} \rightarrow [0,+ \infty)$ as 
$$r_k(z) = \frac{1}{\sum_{(x,h) \in D_k}}\sum_{(x,h) \in D_k,x<z}h$$
which represents the proportion of instances whose feature value $k$ is smaller than $z$.The goal is to find candidate split point $\{s_{k1},s_{k2},...,s_{kl}\}$,such that
$$|r_k(s_{k,j}) - r_k(s_{k,j+1})|<\epsilon ,s_{k1} = min_ix_{ik},s_{kl}=max_ix_{ik}$$
Here $\epsilon$ is an approximation factor. Intuitively, this means that there is roughly $\frac{1}{\epsilon}$ candidate points. Here each data points is weight by $h_i$. To see why $h_i$ represents the weight, we can rewrite Eq() as 
$$ \sum_{i=1}^n\frac{1}{2}h_i(-f_t(x_i) -\frac{g_i}{h_i})^2 + \Omega(f_t) +constant $$
which is exactly weight square loss with label $\frac{g_i}{h_i}$ and weight $h_i$.
根據上述的數學原理，可以知道XGB不是針對樣本個數來定義分位數，而是第二階梯度值，由Eq()得知$h_i$視為加權平方損失的權重，故利用樣本的權重來定義分位數，首先會先定義rank function 來定義特徵值$x<z$的第二階梯度值佔全部第二階梯度值的比例，這個目的是希望相鄰兩個候選切割點之間的第二階梯度差距小於 $\epsilon$，所以說我們可以藉由設定$\epsilon$來得到來選取我們所想要的候選切割點的個數，大約會是 $\frac{1}{\epsilon}$ 候選切割點。
### Sparsity-aware Split Finding
在實際的資料中，$x$通常會是為稀疏(sparse)，以下找能稀疏的可能原因:
* 資料包含遺失值
* 統計上頻率為0的樣本
* 一些特徵工程，像是單熱編碼(One-hot encoding) 

XGB在這些情況給予相對應的解決方案，以遺失值為例，XGB會針對遺失值在每個節點上，增加一個預設的方向，當樣本遺失值時，可以直接給歸類到預設的路徑上，如下圖表示:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/XGBoost/Missing_value.PNG?raw=true)
然而最佳的預測路近可以透過資料學習，在每個內部節點有兩個路近(左或者右)，我們可以列舉該遺失樣本歸類左右分支後的梯度下降量，選擇最大的梯度下降量的路徑設為最優的預設方向，然而此方法只考慮左右兩個分支，故計算量大幅減少，根據作者的實驗，稀疏感知演算法(Sparsity-aware algorithm)比基本演算法速度塊了超過 50 倍，演算過程如下列表示:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/XGBoost/Sparsity-aware%20Split%20Finding.PNG?raw=true)