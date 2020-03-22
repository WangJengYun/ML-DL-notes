---
tags: optimization,gradient descent
---
The Evolution of Gradient Descent 
===
梯度下降法(gradient descent)是一種不斷更新參數的方法，運用在機器學習、深度學習等等，求最佳解的最好的技術，但是也是有想當多限制，由於現今的機器學習的損失函數相當複雜，其實會有很多區域最小值(local minimum)，所以很難找到全域最佳解(global minimum)，因為我們在計算斜率，也就是找到一的方向往最佳解的方向，有很多情況明明不是全域最佳解，但他的斜率是等於0，很難再繼續迭代找到更好的值，除了local minimum，也會有plateau與saddle導致情況，如下圖，接著我們介紹幾種方法避免在迭代過程中，落入斜率等於0這幾個點。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/optimization/The%20Evolution%20of%20Gradient%20Descent/Limitation%20of%20GD.PNG?raw=true)
## Momentum 
Momentum顧名思義就是運動量的意思，主要是藉由物理的運量的概念，讓在同方向的維度上學習速度會變數，方向改變時則會速度會變慢，但會主要的效益是在斜率等於0的點，可以藉由先前的所累積的動能，使他可以輕易跳脫該點，數學公式如下:
$$
\begin{aligned}
&v_t=\gamma v_{t-1} + \eta J(\theta)\\
& \theta_{t+1}=\theta_{t} - v_t
\end{aligned}
$$
其中$\gamma$為momentum，通常設定為0.9或者更小，另外$\theta$為learning rate 
這裡會發現多了$v_t$參數，可以視為方向速度，在一定程度上保持之前的更新方向，然而如果當前的梯度方現與更新方向是一致，則$|v_t|$則會越來越大，代表使梯度值會增強，反之如果方向不同時，$|v_t|$則會比上次更小，另外如果接近最佳解的時候，則梯度值近乎於0，並也可以藉由$\gamma$來將減緩方向速度，可以想像成空氣阻力或者摩擦力等等，通常是設定小於1，最後我們藉由這個方法快速收練，且沒使用Momentum的SGD相比之下可以減少震盪，如下圖:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/optimization/The%20Evolution%20of%20Gradient%20Descent/SGD%20with%20momentum.PNG?raw=true)
## Nesterov's Accelerated Gradient 
Momentum的方法，可以保持更新方向，以利跳脫local minimum，但是盲目更從梯度值，這其實並不穩定，如果以SGD為例的話，如果當前參數值接近最佳值，但是如果抽到樣本是異常情況，給予很大的梯度值，是容易跳開最佳範圍，所以希望有一個聰明的機制，在更新之前，知道目前是要放慢更新速度。所以提出Nesterov's Accelerated Gradient，類似於提前預知梯度值的變化，數學公式如下:
$$
\begin{aligned}
&v_t=\gamma v_{t-1} + \eta J(\theta_t - \gamma v_{t-1})\\
& \theta_{t+1}=\theta_{t} - v_t
\end{aligned}
$$
在更新參數前，我們可以先計算$\theta_t - \gamma v_{t-1}$來預估下一個參數值的位置，這樣更穩定的更新參數值，由下圖可以了解與Momentum差異，你會發現Momentum會試盲目跟從當前的斜率，如果當前的斜率與下一個斜率差異較大時，會產生震盪，然而Nesterov's Accelerated Gradient則會先預估下一個參數值得位置，再進行更新，這會更精準更新至正確參數值，也就是說會根據他們梯度值的重要程度來決定是較大幅度或者小幅度的更新，會使收斂更加的穩定且會更減少震盪的程度。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/optimization/The%20Evolution%20of%20Gradient%20Descent/the%20difference%20betweent%20momentum%20and%20NAG.PNG?raw=true)
