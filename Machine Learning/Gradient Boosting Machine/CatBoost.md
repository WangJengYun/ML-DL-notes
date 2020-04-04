---
tags: Boosting,Machine Learning
---
Catboost 
===
Catboost在2017年由俄羅斯最大搜尋引擎Yandex所發表的演算法，與之前Boost系列的差異，就是添加資料預處理，主要是類別行特徵處理，另外也針對Boost系列的缺點進行改善，如overfitting、預測偏差等等，相對較穩健，在Catboot的paper中，所提出的benchmarks中，模型精準度都比lightGBM與XGB較佳。
## Categorical feature 
類別行特徵是一個數值的離散集合，每個數值之間是無法進行比較，通常是無法放入模型進行訓練，在一般常用的處理方式為單熱編碼(one-hot encoding)，就是將每個離散數值轉換成二元特徵，也就是說數值只有1或0，然而處理方式會有缺點，如果說離散數值個數較少，是可以將她轉換成二元特徵，但是離散數值較多的話，像是use_id等等，此案例我們稱為高基數(High-Cardinality)，採用單熱編碼，則會增加許多維度，通常解決這議題，會採用下列方法:
* 將所有的離散數值進行分組，再採用單熱編碼
* 使用目標統計量方法(target statistics)進行估計期望目標值

catboost演算法是採用目標統計量的方法對類別特徵進行處理，然而在其他boost系列演算法是採用其他方式，像是lightGBM則在每次迭代對每個數值計算梯度統計量，雖然這很重要的訊息，但是計算量與暫存記憶體消耗則會增加很多，因此catboost演算法在模型訓練前，採用目標統計量將離散特徵量化成數值，是最有效且不會失去太多訊息量。接著我們介紹不同的目標統計量的方法，在此之前進行假設，如下:
Assume that dealing with a category feature $i$ is to substitute the category $x^i_k$ of $k$th training example with one numeric feature equal to some target statistic(TS) $\hat{x}^i_k$.commonly, it estimates the expected target $y$ conditioned by the category:
$$\hat{x}^i_k \thickapprox \mathbb {E}(y|x^i=x^i_k)$$

### Greedy TS
A straightforward approach is to estimate $\mathbb{E}(y|x^i=x^i_k)$ as average value of $y$ over the training examples with the same category $x^i_k$. The estimate is noisy for low-frequency categories, and one usually smoothes it by some piror $p$:
$$\hat{x}^i_k = \frac{\sum^n_{j=1}\mathbb{I}_{\{x^i_j=x^i_k\}} \cdot y_j + ap}{\sum^n_{j=1} \mathbb{I}_{\{x_j^i = x^i_k\}}+a}$$
where $a>0$ is a parameter. A common setting for $p$ is the average value in dataset 

Greedy TS 主要是根據各類別的平均目標比例來量化類別變數，但是會造成目標洩漏(target leakage)，也就是很容易overfitting，導致訓練與測試分布差異非常大，然而我們可以找一個極端的例子來說明，假設第$i$個類別變數，其中類別值A都是唯一值，對於我們分類任務可以給定$P(y=1|x^i=A)=0.5$，我們可以透過Greedy TS公式且使用訓練集可以得到$\hat{x}^i_k=\frac{y_k+ap}{1+a}$，然而我們只要使用threshod $t=\frac{0.5+ap}{1+a}$就可以完美分類，但是對於測試集來說Greedy TS為$p$，$p$ 為事件發生的機率，可由整個資料集計算事件發生的比例，如果$p<t$，則為1，反之為0，因此住兩總的準確度都為0.5，但是回違反下列性質:
$$P1\,\,\,\, \mathbb{E}(\hat{x}^i|y=v)=\mathbb{E}(\hat{x}^i_k|y_k=v)$$
在上述的例子為$\mathbb{E}(\hat{x}^i_k|y_k)=\frac{y_k+ap}{1+a}$ 和 $\mathbb{E}(\hat{x}^i|y)=p$，無法滿足上條件，為了避免條件偏移(conditional shift)，所以在計算TS值，我們不使全部的訓練資料而是用部分資料進行計算，就接下來介紹的兩個方法。
### Holdout TS
Assume we partition the training dataset into parts $\mathbb{D} = \hat{\mathbb{D}}_0 \cup \hat{\mathbb{D}}_1$ and use $\mathbb{D}_k=\hat{\mathbb{D}}_0$ for calculating the TS and $\hat{\mathbb{D}}_1$ for training.
$$\hat{x}^i_k = \frac{\sum_{x_j\in \mathbb{D_k}}\mathbb{I}_{\{x^i_j=x^i_k\}} \cdot y_j + ap}{\sum_{x_j\in \mathbb{D_k}} \mathbb{I}_{\{x_j^i = x^i_k\}}+a}$$
where $x_k \in \hat{\mathbb{D}}_0$，holdout TS can satisfies P1 and it violate the following desired property:
$P2$ Effective usage of all training data for calculating TS features and for learning a model 
### Leave-one-out TS
Assume we take $\mathbb{D}_k=\mathbb{D}\x_k$ for training example $x_k$ and $\mathbb{D}_k=\mathbb{D}$ for test ones
$$\hat{x}^i_k = \frac{\sum_{x_j\in \mathbb{D_k}}\mathbb{I}_{\{x^i_j=x^i_k\}} \cdot y_j + ap}{\sum_{x_j\in \mathbb{D_k}} \mathbb{I}_{\{x_j^i = x^i_k\}}+a}$$
where $x_k \in \hat{\mathbb{D}}_k$.
事實上，這也無法避免目標洩漏(target leakage)，假設我們考慮$x^i_k = A$時，讓$n^+$表示$y=1$，然後$\hat{x}_k^i=\frac{n^+-y_k+ap}{n-1+a}$，也是可以藉由threshold $t=\frac{n^+-0.5+ap}{n-1+a}$完全分開，但對於測試樣本來說$\hat{x}_k^i=\frac{n^++ap}{n+a}$，這也不滿足$P1$定義。
### Orderd TS 
Assume we perform a random permutation of dataset and for each example we compute average label value for the example with the same category value placed before the given one in the permutation.Let $\sigma = (\sigma_1,\sigma_2,...,\sigma_n)$ be the permutation and $\mathbb{D}_k =\{x_j:\sigma(j)<\sigma(k)\}$,then $x_{\sigma_p,k}$ is substituted with 
$$\hat{x}^i_k = \frac{\sum_{x_j\in \mathbb{D_k}}\mathbb{I}_{\{x^i_{\sigma_j,k}=x^i_{\sigma_p,k}\}} \cdot y_{\sigma_j} + ap}{\sum_{x_j\in \mathbb{D_k}} \mathbb{I}_{\{x_{\sigma_j,k}^i = x^i_{\sigma_p,k}\}}+a}$$
where $\sigma_p = \sigma(k)$ and $\sigma_j = \sigma(j)$,we add a prior value p and a parameter $a>0$,which is the weight of prior.Adding prior is a common practice and it helps to reduce the noise obtained from low-frequency categories.
這個方法是利用隨機排列方法將資料排列，並使用Sliding window方式來計算TS值，更能處理$\hat{x}^i_k$的變異，且可以滿足P1和P2，也同時使用所有的訓練樣本，唯一注意的是，在不同boost的迭代上，事會採用不同排列順序，然而我們也可藉由下列的實驗，表示Orderd TS明顯比其他方法好。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/catboost/comparison%20of%20target%20statistics.PNG?raw=true)
## Order boost

