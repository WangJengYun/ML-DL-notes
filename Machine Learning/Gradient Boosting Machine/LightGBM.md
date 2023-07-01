---
tags: Boosting,Machine Learning
---
LightGBM
===
LightGBM是在2017年Microsoft所提出基於gradient boosting的輕量級演算法，相對於之前所提到的XGBoost的缺點進行優化，是一個高效的gradient boosting，他有下列的優勢:
* 更快的訓練效率
* 低內存使用
* 更好的準確度
* 支持平行運算，可處理大規模數據
* 支持直接使用類別特徵(category features)

接下來我們會針對LightGBM的細節進行介紹
## Gradient-based One-Side Sampling(GOSS)
由於GBDT的準確度高，已成為廣泛被使用的機器學習演算法，但是還使又效率的問題存在，就是對於每一個切割點，都需要遍歷全部資料來計算訊息增益，其複雜度將受到特徵數量與資料筆數的雙重影響，造成在處理大量資料時相當耗時，解決這個問題最直接的方式就是減少特徵數量與資料筆數。

之前所提到Adaboost是根據樣本權重來表示樣本的重要性而進行訓練，則GBDT則是殘差進行訓練，所以Adaboost所提出的採樣模型是無法應用在GBDT中，所以LightGBM所提出GOSS使用梯度值來代替權重，梯度值對於採樣提供非常有用的訊息，假設梯度小，則該樣本的訓練誤差小，是已經經過很好的訓練，反之梯度大，則該樣本訓練誤差大，之後需要再將它納入訓練集進行訓練。

總而言之，GOSS 是一種抽樣的演算法，保留較大梯度值樣本同時對梯度較小的樣本以隨機抽樣的方式來減少計算成本，可以在減少資料量下與學習決策樹的準確度之間取得良好的平衡，演算過程如下:

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Gradient-based%20One-Side%20Sampling.PNG?raw=true)

由上述演算法中，可知道GOSS事先基於梯度的絕對值進行排序，並且選擇前$a\%$的梯度值較大的樣本和剩下樣本隨機抽出$b\%$樣本，使用這兩份樣本當作訓練集，然而在計算訊息增益時，GOSS會透過一個常數$\frac{1-a}{b}$對被抽樣且較小梯度的樣本進行放大，這樣可以更關注訓練不足的樣本，另一方透過乘上權重來防止取樣對原始資料分布造成太大的影響。

最後在這個論文有提到，在建置樹的過程是採用Variance Gain的大小來尋找切割點，然而在樣本數足夠時，透過GOSS的Sampling方式的Estimated Variance Gain 會收斂至 underlying distribution的Variance Gain，詳細的理論可參考論文。


## Split Search Algorithm 
### Histogram-based Algorithm 
在整個DBDT訓練過程中最耗時的部分，是在主要進行決策樹訓練時找尋最佳切割點，在之前XGBoost主要使用預排序(pre-sorted)的方式來處理分割點，雖然這樣的方式相當簡單且較準確，但是也當多的計算量，此方法的split gain需要$O(\#data \times \#feature)$，這其實對於大量資料來說，是需要相當多的時間進行計算，所以LightGBM這裡使採用 Histogram-based algorithm ，然而這個方法在XGB也可以被選擇，演算過程如下:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Histogram-based%20algorithm.PNG?raw=true)
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Histogram-based%20algorithm_detail.PNG?raw=true)


由上敘的演算法得知，Histogram-based Algorithm 使連續型的特徵離散化，也就是將數值進行分段並分別裝箱裝箱，其實是在畫直方圖的前處理，另外此過程不需要事先排序，因為在分段時，就已經定義數值之間的順序，最後在不同箱子(bins)尋找最佳切割點，在split gain的計算量會減少成 $O(\#bin \times  \#feature)$，另外也會減暫存記憶體。

另外此方法會有一個疑慮，雖然我們進行分段，讓計算量減少，但是對於訊息量相對也會減少，所以有可能會對結果產生影響，但是在實驗數據上，並無差異，離散化分裂對於最終的結果影響並不大，甚至會更好，因為決策樹(decision tree)本來就是弱學習器(weak learner)，然而採用離散化，則會有正規化的效果，有效避免overfiting。

最後我們記錄每一個bin的累加一階梯度值、二階梯度值與樣本各個數，來計算增益，並以它當作基準，找到最大增益進行切割，此過程類似之前XGB的準則，視意圖如下:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/histogram_gain.PNG?raw=true)

### Tree growth algorithm 
接下來我們探討樹的生長策略，通常會有下列兩總策略:
* Level-wise tree growth : 大多決策樹都採用這個策略，基於層進行生長，直到達到停止條件
* Leaf-wise tree growth : LightGBM採用此策略，每次分割增益最大的葉子節點，直到達到停止條件。

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Tree%20growth%20algorithm.PNG?raw=true)

之前所提到XGB是採用Level-wise生長策略，是能夠同時分割同一層的節點，可進行多線程優化，不容易overfitting，但是實際上有很多葉子的分裂分裂增益較低，是根本沒必要進行搜索與分割。

LightGBM則是採用leaf-wise生長策略，每次分割時都會尋找增益最大的一個葉子，因此同Level-wise生長策略相比，再分裂數目相同時，Leaf-wise 是可以降低更多的loss，得到更好的準確度，但是在資料量小時，會容易overfitting，所以LightGBM增加最大深度的限制，保證在高效率的同時可以防止overfitting。

## Optimal Split for Categorical Feature 
之前XGB是不支援類別特徵(category feature)的處理，在這樣面通常我們是會將他使用單熱編碼(one-hot encoding)，但決策樹(decision tree)會有一些缺點如下:
* 不平衡的問題 : 當每個類別的個數有明顯的差異，則切分點增益會是分常小，換句話說，當類別之間有個數是有懸殊差異情形，即使增益很大，但乘上該比例之後幾乎可以忽略，較大的集合，其實就大約等於原始的樣本集，增益幾乎為零。
* 影響決策學習 : 當我們有較多的類別時，藉由單熱編碼(one-hot encoding)轉換，會產生較多的衍生變數，這樣學習效果是彙編差，因為該特徵的預測能力被拆分成很多份，每一份與其他特徵競爭最佳分割點時，最終該特徵得到的重要程度會比實際的較低。

基於上述的問題，LightGBM則是採用many to many的方式進行切割，也就是將所有的類別分為兩集合進行切割，如圖左，另外圖右為單熱編碼的分割。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Optimal%20Split%20for%20Categorical%20Feature.PNG?raw=true)
在每次切割Lightgbm會根據訓練目標對類別進行排序，換句話說，我們是根據累積梯度值$\frac{\sum gradient }{\sum hessian}$對直方圖進行排序，然後在排序的直方圖找到最佳的切割點，視意圖如下:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Optimal%20Split%20for%20Categorical%20Feature_2.PNG?raw=true)
## Exclusive Feature Bundling(EFB)
在面臨大量資料，之前我提到GOSS方法抽出重要的樣本進行訓練，然而來有另外一個議題就是為高維度資料，高維度資料通常為稀疏矩陣，也就是許多為0值，而在稀疏的特徵空間中，許多特徵彼此是互斥(mutually exclusive)，此互斥的定為不會同時有非零值，也就是2個之中會有一個為零值，類似於one-hot encoding，我們可將這些互斥(mutually exclusive)的特徵進行合併，也不會喪失原始資訊，所以LightGBM基於這個概念所採用EFB方法，利用bundle的方式將互斥的特徵進行合併，從而減少特徵數量，以利更又效率的訓練模型，模型複雜度減少至$O(\#data,\#bundle)$，其中$\#bundle << \#features$。

在Exclusive Feature Bundling方法中採取兩個演算法，如下
* Greedy Bundling : 決定將那些特徵合併的一個bundle。
* Merge Exclusive Features : 將bundle內的特徵合併新的特徵

演算法如下:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/EFB_1.PNG?raw=true)
### Greedy Bundling
將互斥(mutually exclusive)的特徵合併是為NP-hard問題，然而這也使一個圖著色問題(Graph Coloring Problem)，給定一個無向圖$G = (V,E)$，其中$V$為頂點集合，$E$為邊集合，圖著色問題是將$V$分成$K$個顏色圖組，每個組形成一個獨立集，集其中沒有相鄰的頂點，最終希望獲得最小值K值，如下圖唯一範例:

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Graph%20Coloring%20Problem.PNG?raw=true)

然而我們可將$G$中的$V$點是為特徵，共有$|V|$個特徵，且特徵之間的總衝突看做是圖中的邊$E$，也就是如果兩個特徵沒有互斥(mutually exclusive)則增加邊，進而合併特徵且使得合併的bundle個數使得最小，通常只有少量的特徵之間並非100%互斥，但絕大多數清況下並不會同時有非零的情況。若建構bundle的演算法允許小的衝突，就可以得到更少數的bundle，進一步提高效率，然而隨機汙染一部分特徵最多影響精準度$O([(1 - \gamma)n]^{(-\frac{2}{3})})$，其中$\gamma$為最大特徵衝突率，如果$\gamma$極小，就可以在效率與準確度之間達到平衡。
然而Greedy Bundling演算過程說明如下:
1. 先建構圖，其中每個邊有權重是來衡量特徵之間的衝突值，所謂的衝突是指在兩個特徵之間計算有重疊非零的個數，
2. 藉由度數進行排序(降序)，度數越大則非零值越多，跟其他特徵的衝突越大。
3. 根據有序的特徵清單進行迭代，分配特徵至已存在的bundle(如果總衝突值<K)，或者建立新的bundling(總衝突值>K)

此演算法的計算量為$O(\#feature)$，然而當特徵數很大時，仍然效率不高，為了改善這個情況，LightGBM提供高有效率且不需要建立圖，就是根據非零的各數進行排序，就是類似於圖節點的度排序，因此更多非零值導致衝突，這只是替代上述的演算法中排序策略而已。

### Merge Exclusive Features
我們已經將features根據衝突值進行分組成bundle，接這進行合併。然而為了保持bundle內的features的互斥，則是將features的範圍擴大，如下為示意圖，原本feaute1範圍為1至4與feature2範圍1至2，我們需要衍生新的特徵將這兩個數值都能考慮，則將新的特徵範圍擴展成1至6，然而其中5和6是對應feautre的1與2。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Merge%20Exclusive%20Features_1.PNG?raw=true)
演算法過程如下:
1. 計算在bundle的每個特徵的bin的數量(或者偏移量)，增加至totalbin，計算紀錄累加的totalbin為binRanges。
2. 建立新的bin，且bin的長度為資料筆數，起始值都設為0
3. 對每筆資料與特徵進行迭代，如果值為0則不會對新的bin進行更新，非0則是原始特徵的bin增加相對應的偏移量。

採用了EFB方法之後，從原本的特徵數減少至bundle個數，進而有小減少數據大小，且可以避免計算特徵內0值，也可以優化histogram-based演算法，可以直接忽略特徵內的0值，再掃描資料的成本也會從$O(\#data)$至$O(\# non\_zero\_data)$，整理來說提高模型的訓練速度。