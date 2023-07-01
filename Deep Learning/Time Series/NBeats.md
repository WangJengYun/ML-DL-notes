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
在block內部包含兩個部分，第一個是「mutiple-layer FC network with Relu nonlinearities」，其輸出/預測基本的擴張係數「basis expansion cofficients」為forwoard的$\theta^f_l$及backward的$\theta^b_l$，第二個是backward $g^b_l$ 及 forward $g^f_l$兩個函示，主要接收各別的$\theta^f_l$及$\theta^b_l$擴張係數，藉由這兩個基本函式映射且產生$\hat{x_l}$及$\hat{y_l}$，示意圖如下，另外其相對應的數學函示如下：
The operation of the first part of the $l$-th block is described by the following equations:
$$h_{l,1} = FC_{l,1}(x_l);h_{l,2} = FC_{l,2}(h_{l,1});h_{l,3} = FC_{l,3}(h_{l,2});h_{l,4} = FC_{l,4}(h_{l,3})$$
$$\theta^b_{l}=LINEAR^b_{l}(h_l,4);\theta^f_{l}=LINEAR^f_{l}(h_l,4)$$
Here LINEAT layer is simply a linear projection layer, i.e. $\theta^f_l=W^f_lh_{l,4}$ and the FC layer is a standard fully connected layer with RELU non-linearity, i.e $h_{l,1} = FC_{l,1}(x_l) = RELU(W_{l,1}x_l+b_{l,1})$.

另外此架構預測$\theta^f_l$，可以藉由$g^f_l$的基礎向量合併，提升部分預測值$\hat{y_l}$精準程度，另外也能預測$\theta^b_l$，可以藉由$g^b_l$來幫助下游的block，移除針對預測沒有幫助的成分，其中$g^f_l$與$g^b_l$可以反應特定問題歸納偏差，有限制的輸出結構，相關數學公式如下：

The second part of the network maps expansion coefficients $\theta^f_l$ and $\theta^b_l$ to outputs via basic layers, $\hat{y_l} = g^f_l(\theta^f_l)$ and $\theta{x_l} = g^b_l(\theta^b_l)$.Its operation is describle by the following equations:
$$\hat{y_l} = \sum^{dim(\theta^f_l)}_{i=1} \theta^f_{l,i}v^f_i\  , \hat{x_l} = \sum^{dim(\theta^b_l)}_{i=1} \theta^b_{l,i}v^b_i$$
Here $v_i^f$ and $v^b_i$ are forecast and backcast basis vectors, $\theta^f_{l,1}$ is the $i$-th element of $\theta^f_l$. The function of $g^b_l$ and $g^f_l$ is to provide sufficiently risk set $\{v_i^f\}^{dim(\theta^f_l)}_{i=1}$ and $\{v_i^b\}^{dim(\theta^b_l)}_{i=1}$ such that their respective outputs can be represented adequately via varying expansion coefficients $\theta^f_l$ and $\theta^b_l$.

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Time%20Series/NBeats/NBeats_2.png?raw=true)
### Doubly Residual Stacking
在過去深度學習方式是較難提供模型結果的解釋，故作者提供新穎方法「hierarchical doubly resuidualtopology」，此方法會有兩個殘差值，一個為backcast prediction 與 forecast prediction，其數學公式如下：
Its operation is describle by the following equations:
$$x_l = x_{l-1}-\hat{x}_{l-1}\  , \hat(y)=\sum_l\hat{y}_l$$

由上可以知道第一個block的輸入值近似於$x$ i.e $x_1 \equiv x$, 另外剩餘的block，所學習的對象是減掉前一個輸出值$\hat{x}_{l-1}$，這樣的機制是會讓下游的block比較好進行預測的任務。更重要的事，每個block會預測出部分$\hat{y}_l$，並在stack層進行聚集，此機制是很好提供階層分解，讓最後的預估的$\hat{y}$是所得部分的$\hat{y}_l$合併。最後我們可以透過隨機$g^b_l$與$g^f_l$，讓梯度更加的的透明，作者也強制將$g^b_l$與$g^f_l$共享給stack中的任一個block，是可以讓模型有解釋的重要部分。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Time%20Series/NBeats/NBeats_3.png?raw=truex)
### Interpretablily
接著針對模型解釋我們提供兩個架構，一個基本的通用的DL，另外一個為明確的歸納偏差進行進行解釋，如下：
#### Generic architecture
此方式不依賴時間序列的概念，而使採用$g^b_l$及$g^f_l$為線性映射，數學公式如下：
In this case the outputs of block $l$ are described as :
$$\hat{y}_l=V^f_l\theta^f_l\  , \hat{x}_l=V^b_l\theta^b_l + b^b_l$$
我們可以透過$V^f_l$學習部分預測$\hat{y}_l$的預測分解，其維度為$H \times dim(\theta^f_l)$，其中第一分解矩陣維度可以表示離散時間指標，第一分解矩陣維度可以表示基本函式的指標，換句話說，可以將$V^f_l$視為時間上波形，但是無法強迫學習時間上的特徵，沒有固有的型態，在實驗中沒有特別明顯，故無法成為任何解釋。
#### Interpretable architecture 
此方式可以透過在Stack level增加架構到基本layer層，此外對於時間序列的領域來說，我們要解釋這個序列，通常會將其分解成趨勢(Trend)及季節性(Seasonality)，故作者使用這個概念到模型中，讓我們Stack層的輸出更有解釋性，若要在Stack具有解釋性，則不會採用generic model，而是下列Trend跟Seasonality的model，說明如下：

**Trend model**
時間趨勢是在時間序列中最典型的特徵，主要是monotonic function或者是a slowly varying function。為了要模擬這個趨勢，故作者建議下列的數學公式:
we propose to constrain $g^b_{s,l}$ and $g^f_{s,l}$ to be a polynomial of small degree $p$, a function slowly varying across forecast window:
$$\hat{y}_{s,l}=\sum^p_{i=0}\theta^f_{s,l,i}t^i$$
Here time vector $t = [0,1,2, ... , H-2, H-1]/H$ is defined on a discrete grid running from 0 to $\frac{H-1}{H}$, forecasting $H$ steps ahead. Alternatively the trend forecast in matrix from will then be:
$$\hat{t}^{tr}_{s,l} = T\theta^f_{s,l}$$
where $\theta^f_{s,l}$ are polynomial coefficients predicted by a FC network of layer $l$ of stack $s$ described by equations. and $T = [1, t, ..., t^p]$ is matrix of powers of $t$. If $p$ is low, e.g. 2 or 3. it forces $\hat{y}^{tr}_{s,l}$ to mimic trend.
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Time%20Series/NBeats/NBeats_4.png?raw=true)

**Seasonality model**
另外一個季節性的趨勢也是蠻典型的特徵，是屬於常規、週期性、常發生波動，因次作者建議下列數學公式:
we propose to constrain $g^b_{s,l}$ and $g^f_{s,l}$ to belong to the class of periodic functions, i.e $y_t = y_{t-\triangle}$, where $\triangle$ is a seasonality period. A natural choice for the basis to model periodic fucniton is the Fourier series:
$$\hat{y}_{s,l} = \sum^{[H/2-1]}_{t=0}\theta^f_{s,l,i}\cos(2\pi it)+\theta^f_{s,l,i+{H/2}}\sin(2 \pi it)$$
The seasonality forecast will then have the matrix form as follows:
$$\hat{y}_{s,l}^{seas} = S \theta^f_{s,l}$$
where $\theta^{f}_{s,l}$ are Fourier coefficients predicted by a FC network of layer $l$ of stack $s$ described by equations. and $S=[1, \cos(2\pi t), ..., \cos(2 \pi [H/2-1]t, \sin(2 \pi t), ..., \sin( 2 \pi [H/2-1]t))]$ is the matrix of sinusoidal waveforms. The forecast $\hat{y}^{seas}_{s,l}$ is then a periodic function mimicking typical seasonal patterns
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Time%20Series/NBeats/NBeats_5.png?raw=true)

最後說明一下如何使用這兩個可解釋架構，主要會將趨勢性及季節性放在各別的Stack，首先會先經過趨勢性的Stack，再經過季節性的Stack，透過雙重殘差方式疊加，會有幾個重要的特性:
1. 在進入季節性的stack之前，可透過趨勢性的stack將輸入$x$的趨勢性進行移除。
2. 部分的趨勢性及季節性可以能夠個別解釋預測情況

最後從結構來看，每一個stock是由殘差連接的不同block所組成的，每個block都可以共享各自的不可以學習的$g^b_{s,l}$與$g^f_{s,l}$，若趨勢性及季節性的block為3時，發現在共享$g^b_{s,l}$與$g^f_{s,l}$為基礎，在不同的block共享所有的權重，是可以或的較好驗證表現。

### Ensembling 
最後作者採用Ensembling方式，將不同的實驗進行聚合，如下:
1. 不同模型去配似三種的評估指標，如 sMAPE、MASE及MAPE
2. 採用不同的window size 進行訓練，如$2H,3H, ..., 7H$，共有6種window size
3. 不同初始化的模型進行訓練

以上共有180種的模型結果，進行中位數的聚合，得到最終預測值，相關的效度，可以參考下列圖表，其中N-BEATS-G表示通用結構; N-BEATS-I 表示可解釋的結構; N-BEATS-I+G 表示聚合通用結構及解釋的結構，可以發現在M4的資料集上，N-BEATS-I+G 表現相對比較好。

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Time%20Series/NBeats/NBeats_6.png?raw=true)
### Experimental Result
由實驗可以知道解釋性的N-BEATS相對比通用性更容易將序列資訊萃取出來。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Time%20Series/NBeats/NBeats_7.png?raw=true)
