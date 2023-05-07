---
tags : Time Series, Deep Learning
---
NBeats
===
時間序列一直是商業上重要命題，過去都是以統計方式來解決，像是在M4 competition的冠軍是採用「neural residual/attention dilated LSTM stack」與「classical Holt-Winters statistical model」做結合，故作者想說是否可以存粹以DL進行模型預測，然而作者提出「backward」和「forward」的殘差進行連結，且此方式也具有解釋程度，此方法相比M4的冠軍方法，提升4%，想當在時間序列領域上有的一大的進步。

## NBeats
Nbeats的全名為「Neural Basis Expansion Analysis For Interpretable Time Series Forecasting」，有兩個個屬性 1.基本架構為簡單且通用，並非複雜。2.此架構不依賴傳統的時間序列工程，最後此架構也能同步探索解釋程度，此方法架構如下：
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Time%20Series/NBeats/NBeats_1.png?raw=true)
接著後續我們針對$l_{th}$ block進行說明，在$l_{th}$ block中，$x_l$ 表示所有此block的輸入，是我們所觀察的過去時間序列，另外輸出為$H$個預測值，我們針對$x$的長度可以是$2H$至$7H$之間，另外其輸入$x_l$是上一個block輸入及輸出的殘差值，每個block都會有兩個輸出，1. $\hat{y_l}$表示為$H$長度的forward預測值 2. $\hat{x_l}$表示$x_l$的最好估計值，可稱為$backward$的預測值，最後我們可以使用限制函數空間近似一些patterns或者signals。

### Basic Block
在block內部包含兩個部分，第一個是「mutiple-layer FC network with Relu nonlinearities」，其輸出/預測基本的擴張係數「basis expansion cofficients」為forwoard的$\theta^f_l$及backward的$\theta^b_l$，第二個是backward $g^b_l$ 及 forward $g^f_l$兩個函示，主要接收各別的$\theta^f_l$及$\theta^b_l$擴張係數，藉由這兩個基本函式映射且產生$\hat{x_l}$及$\hat{y_l}$，其相對應的數學函示如下：
The operation of the first part of the $l$-th block is described by the following equations:
$$h_{l,1} = FC_{l,1}(x_l);h_{l,2} = FC_{l,2}(h_{l,1});h_{l,3} = FC_{l,3}(h_{l,2});h_{l,4} = FC_{l,4}(h_{l,3})$$
$$\theta^b_{l}=LINEAR^b_{l}(h_l,4);\theta^f_{l}=LINEAR^f_{l}(h_l,4)$$
Here LINEAT layer is simply a linear projection layer, i.e. $\theta^f_l=W^f_lh_{l,4}$ and the FC layer is a standard fully connected layer with RELU non-linearity, i.e $h_{l,1} = FC_{l,1}(x_l) = RELU(W_{l,1}x_l+b_{l,1})$.

另外此架構預測$\theta^f_l$，可以藉由$g^f_l$的基礎向量合併，提升部分預測值$\hat{y_l}$精準程度，另外也能預測$\theta^b_l$，可以藉由$g^b_l$來幫助下游的block，移除針對預測沒有幫助的成分，其中$g^f_l$與$g^b_l$可以反應特定問題歸納偏差，有限制的輸出結構，相關數學公式如下：

The second part of the network maps expansion coefficients $\theta^f_l$ and $\theta^b_l$ to outputs via basic layers, $\hat{y_l} = g^f_l(\theta^f_l)$ and $\theta{x_l} = g^b_l(\theta^b_l)$.Its operation is describle by the following equations:
$$\hat{y_l} = \sum^{dim(\theta^f_l)}_{i=1} \theta^f_{l,i}v^f_i\  , \hat{x_l} = \sum^{dim(\theta^b_l)}_{i=1} \theta^b_{l,i}v^b_i$$
Here $v_i^f$ and $v^b_i$ are forecast and backcast basis vectors, $\theta^f_{l,1}$ is the $i$-th element of $\theta^f_l$. The function of $g^b_l$ and $g^f_l$ is to provide sufficiently risk set $\{v_i^f\}^{dim(\theta^f_l)}_{i=1}$ and $\{v_i^b\}^{dim(\theta^b_l)}_{i=1}$ such that their respective outputs can be represented adequately via varying expansion coefficients $\theta^f_l$ and $\theta^b_l$.
### Doubly Residual Stacking
在過去深度學習方式是較難提供模型結果的解釋，故作者提供新穎方法「hierarchical doubly resuidualtopology」，此方法會有兩個殘差值，一個為backcast prediction 與 forecast prediction，其數學公式如下：
Its operation is describle by the following equations:
$$x_l = x_{l-1}-\hat{x}_{l-1}\  , \hat(y)=\sum_l\hat{y}_l$$

由上可以知道第一個block的輸入值近似於$x$ i.e $x_1 \equiv x$, 另外剩餘的block，所學習的對象是減掉前一個輸出值$\hat{x}_{l-1}$，這樣的機制是會讓下游的block比較好進行預測的任務。更重要的事，每個block會預測出部分$\hat{y}_l$，並在stack層進行聚集，此機制是很好提供階層分解，讓最後的預估的$\hat{y}$是所得部分的$\hat{y}_l$合併。最後我們可以透過隨機$g^b_l$與$g^f_l$，讓梯度更加的的透明，作者也強制將$g^b_l$與$g^f_l$共享給stack中的任一個block，是可以讓模型有解釋的重要部分。
### Interpretablily
接著針對模型解釋我們提供兩個架構，一個基本的通用的DL，另外一個為明確的歸納偏差進行進行解釋，如下：
#### Generic architecture
此方式不依賴時間序列的概念，而使採用$g^b_l$及$g^f_l$為線性映射，數學公式如下：
In this case the outputs of block $l$ are described as :
$$\hat{y}_l=V^f_l\theta^f_l\  , \hat{x}_l=V^b_l\theta^b_l + b^b_l$$
我們可以透過$V^f_l$學習部分預測$\hat{y}_l$的預測分解，其維度為$H \times dim(\theta^f_l)$，其中第一分解矩陣維度可以表示離散時間指標，第一分解矩陣維度可以表示基本函式的指標，換句話說，可以將$V^f_l$視為時間上波形，但是無法強迫學習時間上的特徵，沒有固有的型態，在實驗中沒有特別明顯，故無法成為任何解釋。
#### Interpretable architecture 
