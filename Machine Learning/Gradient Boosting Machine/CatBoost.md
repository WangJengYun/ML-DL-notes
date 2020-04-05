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
## Prediction shift and Order boost
### Prediction shift
在還沒有進入Catboost的核心演算法之前，先來談談Gradient Boost的缺點，在catboost論文中，提到預測偏差(Prediction shift)，這其實就是一個特別的目標洩漏(target leakage)，間接造成模型overfitting，在這裡提出三的面向:
* 給定$x_k$下，$g^t(x_k,y_k)$的條件分布與測試的條件分布$g^t(x,y)|x$是偏差，其中$g^t(x,y):=\frac{\partial L(y,s)}{\partial s}|_{s=F^{t-1}(x)}$
* 然而我們預測值會有偏差(bias)
* 最終會影響模型的泛化能力(Generalization)

其實在每一次Gradient Boost的迭代，都是採用相同資料進行訓練，並建立當前的模型$F^{t-1}$，然而此條件分配$F^{t-1}(x_k)|x_k$對於測試集是會有偏差，所以我們稱為預測偏差(Prediction shift)。
#### a simple case of a regression task 
這裡我們舉個簡單迴歸實際例子如下:

Assume we use the quadratic loss function $L(y,\hat{y})=(y-\hat{y})^2$ and the negative gradient $-g^t(x_k,y_k)$ can be substituted by the residual function $y_k-F^{t-1}(x_k)$.Now,we have two features $x^1,x^2$ that are i.d.d. Bernoulli random variables with $p=\frac{1}{2}$ and $y=f^*(x)=c_1x^1+c_2+x^2$.Assume we make $N=2$ steps of gradient boosting with decision stumps(tree of depth 1) and step size $\alpha=1$,We obtain a model $F=F^2=h^1+h^2$,where $h^1$ is based on $x^2$ and $h^2$ is based on $x^2$,what is typical for $|c_1|>|C_2|$.the result is as following:
* If two independent samples $D_1$ and $D_2$ of size n are used to estimate $h^1$ and $h^2$ respectively,then $\mathbb{E}_{D_1,D_2}F^2(x)=f^*(x)+O(1/2^n)$ for any $x \in \{0,1\}^2$
* If the same dataset $D=D_1=D_2$ is used for both $h^1$ and $h^2$,then $\mathbb{E}_{D}F^2(x)=f^*(x)-\frac{1}{n-1}c_2(x^2-\frac{1}{2})+O(1/2^n)$ for any $x \in \{0,1\}^2$

這個例子想要表示當我們使用獨立的資料集，所得到訓練模型為不偏估計，然而如果是用相同的資料集，則會有一個偏差項$\frac{1}{n-1}c_2(x^2-\frac{1}{2})$，此偏差項是與$n$有反比關係且與$c_2$有關聯，最後我們藉由這例子證明Gradient Boosting 是有預測偏差的現象。
### Ordered boosting 
Gradient Boosting都會有預測偏移的問題，在Catboost方法提出Ordered boosting，這個想法是我們取一個對所有資料進行隨機排列為$\sigma$，訓練$n$個不同的模型為$M_1,M_2,...,M_n$，其中n為資料筆數，然而在每一次Boosting的迭代中，我們使用前$j$個資料進行訓練，得到$M_{j-1}$模型來獲得第$j$筆樣本的residual，這樣訓練模型與測試不會用同一個資料集，其視意圖如下，但是這是不可行的，因為我們需要訓練$n$個模型，這樣會增加訓練的複雜度與所需要更多的記憶體來完成，然而在Catboost會透過此概念進行修正且下一節會有詳細說明。
<center class="half">
    <img src="https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/catboost/Ordered%20boosting%20image.PNG?raw=true" width="300"/>
    <img src="https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/catboost/Ordered%20boosting%20algorithm.PNG?raw=true" width="300"/>
</center>

#### Practical implementation of ordered boosting
Catboost有兩個方法來解決Ordered boosting的議題，如下:
* Plain:這個方法採用內建的ordered TS方法對類別轉換，轉換後使用標準的GBDT的演算。
* Odered:則是直接對Ordered boosting進行優化。


Catboost演算法會分為兩個階段，定義樹結構與計算葉節點的數值，所以在一開始會先針對訓練集生成$s+1$個獨立隨機續排序，這個排序集合$\sigma_1,\sigma_2,...,\sigma_s$被使用來評估樹結構的分裂，然而固定樹的結構後，會使用$\sigma_0$來計算樹節點的數值，藉著我們在根據這兩個階段做進一步的說明，另外完整的演算法如下:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/catboost/build%20catboost.PNG?raw=true)
##### 定義樹結構
我們以Ordered模式為例，在訓練的過程中，我們維持數個模型$M_{r,j}$，來表示基於排序$\sigma_r$中的前$j$個樣本所學習到的第$i$個當前預測，在第$t$次迭代中，首先會先從$\sigma_1,\sigma_2,...,\sigma_s$集合隨機抽出一個排序$\sigma_r$，來建構一棵樹$T_t$，在建構樹之前，我們會先依照此排序對類別變數計算order TS，接這才會進行訓練。基於$M_{r,j}(i)$，我們可以計算相對應的梯度值為$grad_{r,j}=\frac{\partial L(y_i,s)}{\partial s}|_{s=M_{r,j}(i)}$，當再生成一棵樹的規則時，我們使意cosine similarity來近似於梯度值$G$，並當作衡量指標，然後對於第$i$樣本，其梯度值為$grad_{r,\sigma_r(i)-1}(i)$(如果是Plain，則是$grad_{r,\sigma_r(i)-1}(i)$)，其中在評估每個候選切割點時，對於第$i$個樣本的葉節點數值為平均$p$集合樣本的$grad_{r,\sigma_r(i)-1}$，其中$p$集合樣本為落在於i樣本的相同的節點$leaf_r(i)$上且$\sigma_r(p)<\sigma_r(i)$，注意的是$leaf_r(i)$根據不同的排序而有所不一樣，最後當樹結構$T_t$被建立完成，我們會使用他來提升/更新模型$M_{r^{'},j}$，相同樹結構$T_t$會被用於所有的模型，但是會依據$r^{'}$與$j$的不同設定上的不同的葉節點來更新不同的模型$M_{r^{'},j}$，進一步提升模型的精準性，另外Plain的模式，這是與GBDT演算法類似，對於類別變數會基於$\sigma_1,\sigma_2,...,\sigma_s$得到的TS數值，同時也是會更新不同的模型$M_{r^{'},j}$。
##### 計算葉節點的數值
在我們固定樹的架構後，最後的模型的葉節點的數值會是依照標準GBDT演算法來計算，訓練樣本$i$的葉節點的數值為$leaf_0(i)$，其實就是我們依照$\sigma_0$的排序來計算，另外如果是測試樣本，則是使用全部的訓樣本來進行計算。






