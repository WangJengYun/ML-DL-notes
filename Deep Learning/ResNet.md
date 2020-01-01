# ResNet
-------------
ResNet 為人類在 ImgaeNet 一大突破，主要是解決深度 CNN 模型較難訓練的問題，刷新過去的歷史:

![87851a1c.png](:storage\32160b23-3afa-470f-937a-6e99d4d46bed\87851a1c.png)

由圖可知，過去 CNN 模型做多也只能到幾十層，一直困在給予太多層就會難以訓練，直到2015年提出 Resnet 層數可以到152層，並且準確定可以到5%(人類辨識得到錯誤率)以下，相當大的一個突破，Resnet 主要核心為殘差學習 ( Residual learning )。

### 深度網絡退化議題

在介紹殘差學習之前，先了解為啥加深之後會難以學習，直覺來說 CNN 模型建構太多層之下，且避免  overfitting，CNN 模型會有更複雜的特徵提取，模型的精準度也會提高，但是當時就有一個實驗(如下圖)，你會發現不管在訓練集或者測試集，56層 error 比20層 error 還要高，這個現象稱為**退化** ( degradation )，這個不是因為 overfitting 所造成的，是因為增加較多的 layer 而無法再最佳化，導致較高的訓練誤差，這個問題就是 CNN 模型無法再深的原因。

![0d3c65d0.png](:storage\32160b23-3afa-470f-937a-6e99d4d46bed\0d3c65d0.png)

### 殘差學習( Residual learning )

殘差學習，就是學習輸入與輸出之間的差異，這樣就可以減少訓練誤差。在學習深層的 CNN 模型時，我們可以將它拆分成數個區塊，此區塊由一些layer推疊而成的結構，在各個小區塊學習殘差，這樣可以減少訓練誤差並加深網絡。一開始輸入為 $x$ ，藉由小區塊學習到的特徵，輸出為 $H(x)$ ，現在我們只要學習殘差表示為 $F(x):=H(x)-x$ (原始學習到的特徵等價於 $F(x)+x$ ，由圖可以知道，此做法可以保留原始學習好的特徵，再學習與真實值的差異，這樣至少不會增加訓練誤差，保有一定的準確度。舉個極端的例子，當 $x$ 已接近真實值時，我們再訓練 $F(x)$，則會近於 0，此性質可讓我們訓練較深的網絡，而不失去準確度。殘差結構如下圖所示， $x$ 稱為 identity ，類似一個捷徑連到我們輸出端，稱為 Shortcut Connection。

![a0f1fb3a.png](:storage\32160b23-3afa-470f-937a-6e99d4d46bed\a0f1fb3a.png)

#### Identity Mapping by shortcuts 

大致上料了解 Residual learning，接下來我們探討更詳細的數學公式與運用情境。上圖為一個Residual learning 的區塊，我們可以寫成:
$$y=F(x,{W_i})+x$$
其中，$x$ 與 $y$ 是區塊的輸入與輸出，$F(x,{W_i})$ 為殘差映射( residual mapping )，由上圖可以知道此區塊有兩個layer層，然而可以寫成$F = W_2{\sigma}(W_1x)$，其中${\sigma}$為$ReLU$，所以$F+x$可以由shortcut connection與各元素相加(element-wise addition)，另外此shortcut connection不會增加新的參數與計算的複雜度。

實際建立深度學習架構時，我們會考慮有可能輸入與輸出會是不同的維度，在之前的數學公式$x$與$F$是相同維度，為了解決此問題，我們藉由線性映射(linear project)$W_s$來符合相同的維度:
$$y=F(x,{W_i})+W_sx$$
作者也藉由實驗證明，這個indentity mapping也有足夠的資訊解決退化議題，這個會新增$W_s$參數，主要是用來符合相對應的維度，可以再兩個feature map與channel的相加。

#### ResNet網絡架構

作者採用VGG的架構，在其基礎上進行修改，透過ResNet觀察是否可以將training error再降低，
由下圖可知，主要是用stride為2進行下抽樣，利用global average pooling 替換 fully-connected，另外修改會是依照兩個原則:
* 對於相同的輸出featute map的維度，此layer層會採用相同的filter的數量
* 如果feature map的維度減半，則會將filter的數量增加一倍，這樣可以讓feature map得數量增加一倍，為了保留每一層的網絡複雜度，

由圖進一步了解，實線為輸出與輸入有相同feature map，虛線是維度增加時，我們會採用兩種做法:
* identity mapping 會用zero-padding增加维度，這樣就不會增加參數
* 使用projection shortcut來符合對應的維度(藉由1$\times$1 convolutions)。

對於這兩個方法，都採用stride為2進行下抽

![ebc96065.png](:storage\32160b23-3afa-470f-937a-6e99d4d46bed\ebc96065.png)

##### ResNet-18與ResNet-34的結果
下圖是訓練ImageNet，ResNet使採用zero-padding，則不會增加參數，由圖可知確實ResNet可以減少training error。
![6ff043d9.png](:storage\32160b23-3afa-470f-937a-6e99d4d46bed\6ff043d9.png)

![9eac850a.png](:storage\32160b23-3afa-470f-937a-6e99d4d46bed\9eac850a.png)

#### 深層的ResNet 
接下來建構較深層的ResNet，在這之前我們更深入了解Identity和Project Shortcuts，作者推薦三個projection shortcuts的設定:
* A. zero-pandding shortcuts被使用在增加維度，不會增加任何參數
* B. 對於增加維度，則採用Project Shortcuts，其他則採用indentity 
* C. 不管有沒有稱加維度，都使採用Project Shortcuts

由下表可知道，所有設定都比原始的網絡來的好，其中B稍微比A較好，作者認為A的確較沒有殘差學習，
另外ABC三都只是些微的差距，並沒有對於解決退化議題有明顯的降低訓練誤差。

![ccf32c58.png](:storage\32160b23-3afa-470f-937a-6e99d4d46bed\ccf32c58.png)


針對深層的ResNet，作者採用bottleneck block，由下圖左邊為Block，此輸入的維度與輸出的維度一致，作者將它採用她在ResNet-34，下圖左邊為bottleneck block，對於殘差函式$F$，設計三個layer為一個區塊由1$\times$1、3$\times$3與1$\times$1所構成，其中1$\times$1是主要減少維度，然後再增加維度，這兩個block有相似的模型複雜度。

![03fa6c88.png](:storage\32160b23-3afa-470f-937a-6e99d4d46bed\03fa6c88.png)

由下表可以知道，ResNet-152可將誤差降到4.49%，如果採用ensemble時，誤差可以降到3.57%

![9b284da6.png](:storage\32160b23-3afa-470f-937a-6e99d4d46bed\9b284da6.png)

這個是所有ResNet的模型架構

![52236a2a.png](:storage\32160b23-3afa-470f-937a-6e99d4d46bed\52236a2a.png)

Reference:

* [Deep Residual Learning for Image Recognition(2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)