---
tags : Tips,Deep Learning
---
Batch Normalization
===

## Internal Covariate Shift(ICS)
我先定義ICF是在深層NN在訓練過程中，由於NN的參數變化而導致內部節點的數據分布發生變化，這樣NN的上層需要不斷的適應這些分布變化，使得訓練模型變得更困難。我們以數學來解釋:
首先我們先定義每一層線性轉換為$Z^{[l]} = W^{[l]} \times input +b^{[l]}$，其中$l$表示層數，另外透過激活函數作為非線性轉換為$A^{[l]} = g^{[l]}(Z^{[l]})$，其中$g^{[l]}(.)$為第$l$層的激活函數，接著以梯度下降方法，對每一層的參數 $W^{[l]}$ 與 $b^{[l]}$進行更新，那麼$Z^{[l]}$的分布也就發生變化，進而影響$A^{[l]}$ 作為$l+1$層的輸入，接著$l+1$層就是需要去不停適應這種參數變化，這個過程就稱為Internal Covariate Shift (ICS)。

### 造成的問題
1. 上層NN需要不停調整來適應輸入的數據分布變化，導致NN學習數度降低
2. NN的訓練過程中容易陷入梯度飽和區，減緩NN收斂速度

第一點在上述已經解釋過了，第二也就是在建構NN架構時，採用飽和激活函數(saturated activation function)，如sigmoidc或者tanh，就如霞圖表是，很容易使模型的訓練陷入飽和區(saturated regime)，也就是在訓練過程中，參數$W^{[l]}$會逐漸更新變大或變小，導致$Z^{[l]} = W^{[l]}A^{[l]}+b^{[l]}$就會隨之變大或點小，而且是會受到更上層的NN參數$W^{[1]},W^{[2]},W^{[3]},...,,W^{[l-1]}$的影響，隨者NN加深，則$Z^{[l]}$很容易陷入梯度飽和區，例如sigmoid的兩端，此時梯度會比的變得很小或趨近於0，導致參數無法更新，這也就是梯度消失的議題，所以對於這個問題可以有下列幾的方法處理:
1. 採用非飽和的激活函數，例如:ReLU
2. 採用Normalizarion方法，讓激活函數的數入分布保持在一個穩定的狀態，來避免陷入梯度飽和區。

<center class="half">
    <img src="https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/tips/solve%20the%20problem_1.PNG?raw=true" width="300"/>
    <img src="https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/tips/solve%20the%20problem_2.PNG?raw=true" width="300"/>
</center>

## Whitening
白化(Whitening)使在paper所提出的名詞，是機器學習中，最常使用的一種規範數據分布的方法，轉換之後，進而達到下列兩個目的:
1. 使輸入的分布具有相同的mean與variable
2. 去除特徵之間的相關性

隨然可以藉由Whitening避免ICS的問題，但是也有相對應的缺陷，如下:
1. 由於在每一次迭代訓練都需要做Whitening處理過程，則計算成本太高
2. Whitenin改變了NN的每一層的分布，因而使參數訊息會被Whitenin而丟失調。
## Batch Normalization
為了克服上面的問題，提出Batch Normalization方法，可以讓NN可以更有效率的收斂且克服NN難以訓練的問題，像是梯度爆炸與消失等等。在模型訓練過程中，會是以mini-batch為單位進行正規化，來增加NN的穩定性，也避免使用全訓練集的正規化帶來的高成本，其計算公式如下:
**Input:** 
* Value of $x$ over a mini-batch : $B=\{x_{1,..,m}\}$
* Parameters to be learn: $\gamma$,$\beta$

**Output:**
\begin{align}
&\mu_B \leftarrow \frac{1}{m}\sum^m_{i=1}x_i\tag{mini-batch mean}\\
&\sigma_B^2 \leftarrow \frac{1}{m}\sum^m_{i=1}(x_i-\mu_B)^2\tag{mini-batch variance}\\
&\hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}\tag{normalize}\\
&y_i \leftarrow \gamma\hat{x}_i+\beta \equiv BN_{\gamma,\beta}(x_i)\tag{scale and shift}\\
\end{align}
有上述可以了解，透mini-batch的方法。輸入$m$個樣本來計算平均數與變異數，針對給個維度來標準化，使得分布為平均值為0與變異數為1，使得在經過sigmoid或tanh的激活函數，可以縮放至非線性的激活函數的線區域中，另外使用可學習的$\gamma$與$\beta$，來恢復數據原本的表達能力，且當$\gamma = \sqrt{\sigma}$與$\beta=\mu$時，可以實現等價轉換(identity transform)，保留原始輸入的特徵分布。 
### Batch Normalization的優勢
Batch Normalization在paper實驗中，可以證明能夠解決NN的難以訓練的問題，可以解決的議題如下:
1. **BN使得NN的每層輸入的數據分析的分布可以更穩定，能有效的訓練**

BN在每一層將分布放縮至一定的範圍，將有效的資訊傳遞下去，不必再花時間間去適應底層的輸入的變化，且允許每一層可以獨立學習有效的資訊，有利提高整個NN的學習速度，如下圖小表示差異:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/tips/the%20process%20of%20training%20.PNG?raw=true)

2. **BN較能使用飽和性的激活函數(如:sigmoid,tanh等)，減緩梯度爆炸與消失議題**

在BN處理過程中，使用標準化的方法使數據分布縮放至非線性轉換的的線性區域(非飽和的區域)，能可以得到較高的梯度值，減緩模型難以訓練的議題，最後再使用$\gamma$與$\beta$參數來保留更多的原始資訊。

3. **BN能夠使用較大的學習率(learning rate)**

在訓練NN時，如果使用較大的learning rate，會使得更新步伐過大，容易導致不收斂的情況，然而使用BN可以使NN不受到learning rate影響，如下下列數學式子表達:
For a scalar $a$
$$BN(Wu)=BN((aW)u)$$
and we can show that 
$$\frac{\partial BN((aW)u)}{\partial u}=\frac{\partial BM(Wu)}{\partial u}$$
$$\frac{\partial BN((aW)u)}{\partial aW}=\frac{1}{a}\times \frac{\partial BM(Wu)}{\partial W}$$
其中$u$為前一層的輸入。
我可以看到經過BN的轉換後，權重的縮放值會被去掉，因此保證數據分布可以再穩定的範圍內，另外權重的縮放也部會影像梯度值的計算，有上面的公式表示，當權重越大時，即$a$越大，$\frac{1}{a}$,越小，表示權重$W$的梯度值越小，這樣保證梯度值不會依賴參數的scale，使得參數處於更穩定的狀態，因此BN抑制參數的變化，不會隨著層數加深而被放大或者縮小，此時使用可以設置較大的learning rate而不用擔心不會收斂的問題。

4. **BN據由一定的正規化的效果**

由於在BN中，我們採用mini-batch的樣本，來計算平均數與變異數作為整理的平均數與變異數，然而在不同mini-batch的平均數與變異數也會不同，這樣在NN學習過程中增加一些noise，與dropout帶來的noise滿類似的，所以在一定程度上有可正規化的效果，具有泛化的效果。



