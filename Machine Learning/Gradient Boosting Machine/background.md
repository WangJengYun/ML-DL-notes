---
tags: Machine Learning
---
Background
===
## 前言(Introduction)
Gradient Boosting Machine 目前在結構資料上最常使用的演算法之一，在 Kaggle 上所聽到的常勝軍 XGB、LightGBM 與 Catboost 都是基於這個概念進行調整與創新，本章節針對Gradient Boosting Machine基本架構進行說明，若有敘述不正確請不吝指教。
## 概念(Concept)
接著我們進一步了解Gradient Boosting，稱為梯度提升，是為弱學習模型的機器學習方法，它包含兩種概念，分別為梯度下降(grandient descent)與增強學習(boosting learning)，如下:
### 增強學習(Boosting Learning)
Boosting 的概念是相當簡單，主要透過一群弱學習器(weak Learner)做疊加，會根據前一個弱學習器(weak learner)所預測錯誤的sample在進行訓練，而獲得一個強學習器(strong Learner)可有較高準確率，這個方法主要是用於分類任務，其中最具有代表性的演算法為 Adaboost ，早期解決很多分類問題。
#### Adaboost
Adaboost主要的概念如下圖，為每次迭代的弱學習器(week learner)，所將分類錯誤的樣本的權重放大，這樣可以讓之後訓練每個弱學習會聚焦在錯誤的訓練樣本上，其實就是錯誤中在學習，最後每個弱學習器進行加權投票機制，這樣可以讓準確度要高的弱學習器有較高的權重，反之準確度較低則會有較低的權重，進而增加總體的準確度。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Background/Adaboost1.png?raw=true)

#### Adaboost的演算過程如下:

We have raw input/traning sample $x=(x_1,x_2,...,x_n)\in\mathbb{R}$ with labels $y\in\{-1,1\}$ as is usual in binary classification.The weight of this distribution on traning example $i$ on round $i$ is denoted $w_t(i)$, giving the weak learners $h_t(x_i)$ at the data point $x_i$ and The goodness of a weak hypothesis is measured by its error:
$$ \epsilon_t= Pr_{i∼w_t}(h_t(x_i)\neq y_i)=\frac{\sum_{i:h_t(x_i) \neq y_i}w_t(i)}{\sum^n_{i=1}w_t(i)}\tag{1}$$
##### Pseudocode:
**Input:**<br>
* Given:$(x_1,y_1),(x_2,y_2),...,(x_n,y_n)$ where $x_i\in X,y_i \in Y=\{-1,1\}$<br>
* Initialize $w_1(i) = \frac{1}{n}$

**for** $t\leftarrow 1$ **to** $T$ **do**<br>
* Train weak learner using distribution $w_t$
* Get weak hypothesis $h_t:X\rightarrow \{-1,1\}$ with error$$\epsilon_t=Pr_{i∼w_t}(h_t(x_i)\neq y_i)=\frac{\sum^n_{i=1}w_t(i)I(y_i\neq h_t(x_i))}{\sum^n_{i=1}w_t(i))}\tag{2}$$
* Choose $\alpha_t = \frac{1}{2}\ln(\frac{1-\epsilon_t}{\epsilon_t})\tag{3}$
* Update:\begin{aligned}w_{t+1}(i)&=
\begin{equation}
\left\{
             \begin{array}{lr}
             \frac{w_t(i)e^{-\alpha_t}}{Z_t}\;\;\;\;\;\;if\;h_t(x_i)=y_i\\
              \frac{w_t(i)e^{\alpha_t}}{Z_t}\;\;\;\;\;\;if\;h_t(x_i)\neq y_i
             \end{array}
\right.
\end{equation}
\\ &=\frac{w_t(i)exp(-\alpha_ty_ih_t(x_i))}{Z_t}\\
&=w_1(i)\prod_{t=1}^t\frac{exp(-y_i\alpha_th_t(x_i)))}{Z_t}\\
&=\frac{1}{n}\frac{exp(-y_i\sum_{t=1}^t\alpha_th_t(x_i))}{\prod_{t=1}^tZ_t}
\end{aligned}$$\tag{4}$$
where $Z_t$ is a normalization factor(chosen so that $w_{t+1}$ will be a distribution)

**Ouput:**<br>
$$H(x) = sign(\sum^T_{t=1}\alpha_th_t(x))\tag{5}$$

在Adaboost演算過程中，有個還不錯的特性就是更新權重，在公式(4)可知道當分類錯誤的樣本，其權重會乘於$exp(\alpha_t)$使該筆的樣本的權重變大，反之預測正確的樣本則會乘上$exp(-\alpha_t)$使他們權重變小，接著就會思考所謂$\alpha_t$是如何給定的?因為在我們每個弱學習有好有壞，所以在每次迭代的更新權重會是有差別，則會影響$\alpha_t$值的給定，在此我們可以藉由公式(2)來判斷每個弱學習器的優劣，在此我們會有初步的假設，如果$\epsilon_t>0.5$為較好的弱學習器，反之則較差，好跟壞都有一定的影響程度，推論如下:<br>

* 首先我們先定義更新權重，如果分類錯誤，則會乘上某個量$d_t$，反之分類正確則會除於$d_t$，數學公式如下:
$$
w_{t+1}(i) = 
\begin{equation}
\left\{
\begin{array}{lr}
             \frac{w_t(i)}{d_t}\;\;\;\;\;\;if\;h_t(x_i)=y_i\\
               w_t(i)\times d_t\;\;\;\;\;\;if\;h_t(x_i)\neq y_i
             \end{array}
\right.\tag{6}
\end{equation}
$$
* 如果如 $\epsilon_t = 0.5$，則 $\alpha_t=0$，此弱學習器的訓練是無效的，所以假設 $\epsilon_t = 0.5$ 的情況下進而推論，如下:
$$
\begin{aligned}
&\frac{\sum^n_{i=1}w_t(i)I(y_i\neq h_t(x_i))}{\sum^n_{i=1}w_t(i))} = 0.5 \\
&\Rightarrow \frac{\sum_{i:h_t(x_i) \neq y_i}w_t(i)*d_t}{\sum_{i:h_t(x_i) \neq y_i}w_t(i)*d_t+\sum_{i:h_t(x_i) \neq y_i}\frac{w_t(i)}{d_t}} = 0.5\\
&\Rightarrow \sum_{i:h_t(x_i) \neq y_i}\frac{w_t(i)}{d_t} = \sum_{i:h_t(x_i) \neq y_i}w_t(i)*d_t \\
&\Rightarrow \frac{1}{d_t}(1-\epsilon_t)\sum_iw_t(i)=d_t\epsilon_t\sum_iw_t(i) \\
&\Rightarrow d_t^2 = \frac{(1-\epsilon_t)\sum_iw_t(i) }{\epsilon_t)\sum_iw_t(i)} \\
&\Rightarrow  d_t = \sqrt{\frac{(1-\epsilon_t)}{\epsilon_t}},where\;d_t > 0
\end{aligned}
\tag{7}
$$
* 由公式(7)所推論出來的結果帶入公式(6)，如下:
$$
w_{t+1}(i) = 
\begin{equation}
\left\{
\begin{array}{lr}
             \frac{w_t(i)}{\sqrt{\frac{(1-\epsilon_t)}{\epsilon_t}}}=w_t(i)\times exp(-\frac{1}{2} \ln(\frac{(1-\epsilon_t)}{\epsilon_t}))=w_t(i)e^{-\alpha_t}\;\;\;\;\;\;if\;h_t(x_i)=y_i\\
              w_t(i)\times \sqrt{\frac{(1-\epsilon_t)}{\epsilon_t}}= w_t(i)\times exp(\frac{1}{2} \ln(\frac{(1-\epsilon_t)}{\epsilon_t}))=w_t(i)e^{\alpha_t}\;\;\;\;\;\;if\;h_t(x_i)\neq y_i
             \end{array}
\right.\tag{8}
\end{equation}
$$

由此可證，當 $\epsilon_t = 0.5$ 時，$\alpha_t$ 為 0，則$w_{t+1}(i)=w_t(i)$，此迭代的權重不會進行任何更新。最後我將${h_1(x),h_2(x),...,h_T(x)}$進行加權平均如公式(5)，得到最終的預測結果。
### 梯度下降(Grandient Descent)
梯度下降演算法是目前在機器學習中最常被廣泛使用於優化參數的方式，使用迭代的方式往最佳解方向的進行搜索，在搜索的過程中，不斷更新參數找到最佳解。在機器學習中，我們會先定義損失函數(loss function)或者稱為目標變數(objective function)，我們通常希望某參數值可以使此函數可以逼近最小值，進而找到最好的模型，接著我們會以數學的角度說明:

Let $f(\theta)$ be the objective function which is differentiable and $\theta$ is a finite set of parameters.Now,given a particular point $\theta_0$ and we would like to find the dirtection $d$ s.t. $f(\theta_0+d)$ is minimized as following:
$$f(\theta_0+d)<f(\theta_0)$$
In order to find the steepest direction,we can approximate the loss function via a first-order Taylor expansion as follow:
$$f(\theta_0+d) \approx f(x)+ \bigtriangledown f(\theta_0)d$$
The direction $d$ that minimizes the loss function implies the following optimization problem:
$$min_{d:||d||=1} \bigtriangledown f(\theta_0)d$$

Let we find $d^* \in argmax_{||d||=1} \bigtriangledown f(\theta_0)^Td$.From the Cauchy-Schwarz inequality we know that $\bigtriangledown f(\theta_0)^Td \le ||\bigtriangledown f(\theta_0)||\,||d||$ with equality when $d=\lambda \bigtriangledown f(\theta_0),\lambda \in \mathbb{R}$. Since $||d||=1$ this implies:
$$d^* = \frac{\bigtriangledown f(\theta_0)}{||\bigtriangledown f(\theta_0)||}$$
So,we aim to minimize the function,we seek $\hat{d} \in argmin_{||d||=1} \bigtriangledown f(\theta)$ which is simple $-d^*$,so the iteration of steepest descent in $l_2$ norm are : 
$$\theta^{(t+1)} = \theta^{(t)} - \eta^{(t)} \bigtriangledown f(\theta^{(t)})$$
where $\eta^{(t)}$ is some step size(think about this as some tern mutiplied by $||\bigtriangledown f(\theta^{(t)})||^{-1}$).Notice that this is the gradient descent algorithm.

由上述數學式子說明，我們利用對參數偏微分$(\bigtriangledown f(\theta^{(t)})$，取的此參數的斜率切線，並往負的方向移動，也就是減去負的斜率，並根據 $\eta$ 來決定移動的大小，然而隨著迭代次數，漸漸逼近最佳解，直到沒有明顯改善為止，我們就會認為此解是近似最佳解，視意圖如下:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Background/Grandient%20Descent.png?raw=true)
## Gradient Boosting
我們了解Gradient與Boosting概念之後，來充重新思考Boosting主要由許多弱學習器(weak learner)所組成的強學習器(strong Learner)，每執行一個弱學習器(weak learner)可以使我們的 loss function 降低，這個概念是很類似 gradient descent algorithm，但是我們優化的不是參數，而是模型，就由每次迭代，建立弱學習器來提升模型的的準確度，這種慢慢學習使可以達到不錯表現，準確度也相對其他的模型較高。

Gradient Boosting其實很類似我們之前所提到adaboost，但是有些地方是不一樣的，adaboost會以先前的弱分類器所分錯的樣本調整他的權重，並且會著重這些錯誤的樣本進行訓練，然而 Gradient Boosting 主要演算會是以序列方式增加至先前 underfitting 的預測值，來校正其錯誤，所以主要以先前預測值的殘差進行訓練，最後將所有預測結果累加得到最終的預測值，驗算過程如下:














