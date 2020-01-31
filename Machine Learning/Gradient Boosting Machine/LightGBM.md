---
tags: Machine Learning
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

## Split Search Algorithm 
在整個DBDT訓練過程中最耗時的部分，是在主要進行決策樹訓練時找尋最佳切割點，在之前XGBoost主要使用預排序(pre-sorted)的方式來處理分割點，雖然這樣的方式相當簡單且較準確，但是也當多的計算量，此方法的split gain需要$O(\#data \times \#feature)$，這其實對於大量資料來說，是需要相當多的時間進行計算，所以LightGBM這裡使採用 Histogram-based algorithm ，然而這個方法在XGB也可以被選擇，演算過程如下:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Histogram-based%20algorithm.PNG?raw=true)
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Histogram-based%20algorithm_detail.PNG?raw=true)

由上敘的演算法得知，Histogram-based Algorithm 使連續型的特徵離散化，也就是將數值進行分段並分別裝箱裝箱，其實是在畫直方圖的前處理，另外此過程不需要事先排序，因為在分段時，就已經定義數值之間的順序，最後在不同箱子(bins)尋找最佳切割點，在split gain的計算量會減少成 $O(\#bin \times  \#feature)$，另外也會減暫存記憶體。

另外此方法會有一個疑慮，雖然我們進行分段，讓計算量減少，但是對於訊息量相對也會減少，所以有可能會對結果產生影響，但是在實驗數據上，並無差異，離散化分裂對於最終的結果影響並不大，甚至會更好，因為決策樹(decision tree)本來就是弱學習器(weak learner)，然而採用離散化，則會有正規化的效果，有效避免overfiting。

最後我們記錄每一個bin的累加一階梯度值、二階梯度值與樣本各個數，來計算增益，並以它當作基準，找到最大增益進行切割，此過程類似之前XGB的準則，視意圖如下:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/histogram_gain.PNG?raw=true)

## Optimal Split for Categorical Feature 
之前XGB是不支援類別特徵(category feature)的處理，在這樣面通常我們是會將他使用單熱編碼(one-hot encoding)，但決策樹(decision tree)會有一些缺點如下:
* 不平衡的問題 : 當每個類別的個數有明顯的差異，則切分點增益會是分常小，換句話說，當類別之間有個數是有懸殊差異情形，即使增益很大，但乘上該比例之後幾乎可以忽略，較大的集合，其實就大約等於原始的樣本集，增益幾乎為零。
* 影響決策學習 : 當我們有較多的類別時，藉由單熱編碼(one-hot encoding)轉換，會產生較多的衍生變數，這樣學習效果是彙編差，因為該特徵的預測能力被拆分成很多份，每一份與其他特徵競爭最佳分割點時，最終該特徵得到的重要程度會比實際的較低。

基於上述的問題，LightGBM則是採用many to many的方式進行切割，也就是將所有的類別分為兩集合進行切割，如圖左，另外圖右為單熱編碼的分割。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Lightgbm/Optimal%20Split%20for%20Categorical%20Feature.PNG?raw=true)
在每次切割Lightgbm會根據訓練目標對類別進行排序，換句話說，我們是根據累積梯度值$\frac{\sum gradient }{\sum hessian}$對直方圖進行排序，然後在排序的直方圖找到最佳的切割點。
## Tree growth algorithm 