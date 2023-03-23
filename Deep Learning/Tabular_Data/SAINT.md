---
tags : MLP,Deep Learning
---
ELECTRA
===
過去Tabular Data已經有豐富的應用場景，像是詐欺預測、房價預測等等，此部分大多都是以gradient boosting與random forests演算法為主，處理分類及回歸問題，然而近期也會思考若以深度學習的方法會是如何應用？故本篇想與大家分享SAINT這個模型，主要是藉由2017年google所提出self-attention方法運用在columns與rows，切潛層layer會是以embedding作為轉換，作者有提到此方法是可以超越gradient boosting方法，像似XGBoost、CatBoost與LightGBM。
## Introduction
SAINT的全名為「self-Attention and Intersample Attention Transformer」，透過此方法可以克服一些訓練Tabular Data的困難。主要可將連續行變數及類別行變數映射到向量空間，並視為NLP的tokens當作transformer的輸入，接這transformer中有兩個方式來訓練，如下：
1. self-attention : 此部分主要關注每個資料點的個別的特徵
2. intersample attention：此部分可以藉由特定資料點於其他資料的關係來進行資料的分類，原理會是比較接近nearrest-neighbor classification。

最後此方法針對semi-supervised problem，採用self-supervied contrastive pre-training來增強訓練的結果。
## Self-Attention and Intersample Attention Transformer (SAINT)
Suppose $D=\{x_i,y_i\}^m_{i=1}$ is a tabular dataset with $m$ points, where each $x_i$ is an $n$-dimensional feature vector, and $y_i$ is a label or target variable. Similar to BERT, we append a [CLS] token with a learned embedding to each data sample. Let $x_i = [[CLS], f_i^{\{1\}}, f_i^{\{2\}}, f_i^{\{3\}},..., f_i^{\{n\}}]$ be a single data-point with categorical or continuous features $f_i^{\{j\}}$, and Let $E$ be embedding layer that embeds each feature into a $d$-dimensional space. Note that $E$ use different embedding functions for different features. For a given $x_i \in R^{(n+1)}$, we get $E(x_i) \in R^{(n+1 \times d)}$.
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Tabular_Data/SAINT/SAINT_1.png?raw=true)

### Encoding the Data 
在語言模型當中，所有的token的embeding過程都是相同，但是在tabular的領域中，不同的特徵會來自不同的分配，故需要異質/不同的Eembeddings，另外針對連續型變數，格外透過fully-connected layer with a ReLU nonlinearity進行映射，從$1$維度空間到$d$維度空間。
### Architecture 
由上圖可知，SAINT是由$L$個相同階層所組成，每個階層可分為「self-attention transformer block」與「intersample attention transformer block」，其中self-attention是跟2017年google所發行「Attention is all you need」所使用方法是相同的，第一階層為multi-head self-attention layer(MSA)，接著會連接兩個fully-connected feed-forward(FF) layer with a GELU non-lineariry，其中每個layer都會有skip-connection and layer normalization，另外intersample attention部分是與self-attention相同，只是其中multi-head self-attention layer會是由intersample attention layer(MISA)所替換，此部分會在下個小節說明。

The SAINT pipeline, with a single stage($L$=1) and a batch of $b$ inputs, is described by the following equations. We denote multi-head self-attention by MSA, multi-head intersample attention by MISA, feed-forward layers by FF, and layer norm by LN:
$$\begin{equation} z_i^{(1)}=LN(MSA(E(x_i)))+E(x_i) \end{equation}$$
$$\begin{equation} z_i^{(2)}=LN(FF_1(z_i^{(1)}))+z_i^{(1)} \end{equation}$$
$$\begin{equation} z_i^{(3)}=LN(MISA(z_i^{(2)}\}^{b}_{i=1}))+z_i^{(2)} \end{equation}$$
$$\begin{equation} r_i=LN(FF_2(z_i^{(3)}))+z_i^{(3)} \end{equation}$$
where $r_i$ is SAINT's contextual representation output corresponding to data  point $x_i$. This contextual embedding can be used in downstream tasks such as self-supervision or classification.
### Intersample attention
接著與大家介紹Intersample attention，以每次訓練的batch為主，橫跨不同資料點進行attention計算，首先會先將每個特徵的embedding進行串聯合併，並以不同的資料點進行attention計算，能夠藉由檢視其他資料點來改善資料點的訓練表現，另外若特徵有包含遺失值及雜訊，Intersample attention可以從同個batch中其他資料點參考對應的特徵，能夠讓模型可以精準學習。

如下圖表示，可以透過Intersample attention機制連結不同資料點的資訊，另外作者說明在multi-head中，針對$q$,$k$,$v$不是採用$d$-dimension，而是將這些映射到$\frac{d}{h}$，其中$h$為head的個數，最後我們串連所有的更新向量，得到長度為$d$的向量$v_i$。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Tabular_Data/SAINT/SAINT_2.png?raw=true)

最後作者有提供pseudo-code給大家參考，如下：
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Tabular_Data/SAINT/SAINT_3.png?raw=true)

### Self Supervised pre-training
Contrastive Learning之前在影像與文字有很好的應用，進幾年來開始用運用在tabuler data上，主要概念就是訓練一個降維的網路，並且設計出Contrasive Loss，使得相似的樣本降維後在空間上越近越好，差異大的樣本則盡可能的拉開距離。在上述的訓練過程中，模型會對資料的特徵有所歸納與認識，才能將原始數據去蕪存菁。
#### Generating augmentations
標準的Contrastive Learning其實較難運用在tabular data，VIME的作者提供mixup方法運用在non-embeded space視為data augmentation，但是對於連續行變數會是有限制，主本論文的做作者則是先透過CutMix方法在原始變數上，然後再Mixup方法應用在embedding space，這兩格方法結合起來可以達到有效的self-supervision任務上，數學定義如下：

Assume that only $l$ of $m$ data points are labeled. We denote the embedding layer by $E$, the SAINT network by $S$ and 2 projection heads as $g_1(.)$ and $g_2(.)$. The CutMix augmentation probability is denoted $p_{cutmix}$ and the mixup parameter is $\alpha$. Given point $x_i$, the original embedding is $p_i=E(x_i)$, while the augmented representation is generated as follows:
$$\begin{equation} x^{'}_i = x_i\bigodot m + x_a\bigodot(1-m)\end{equation}$$
$$\begin{equation} p^{'}_i = \alpha * E(x^{'}_i)+(1-\alpha)*E(x^{'}_b)\end{equation}$$
where equation 5 is that cutmix in raw data space and equation 6 is that mixup in embedding space and $x_a$, $x_b$ are random samples from the current batch. $x^{'}_b$ is the CutMix version of $x_b$, $m$ is the binary mask vector sampled from a Bernoulli distribution with probability $p_{cutmix}$ and $\alpha$ is the mixup parameter. 

Note that we first obtain a CutMix version of every data point in a batch by randomly selecting a partner to mix with. We then embed the samples and choose news partners before performing mixup 
#### SAINT and projection heads
現在我們很清楚知道原始的 $p_i$ 和混合 $p^{'}_i$ embeddings，透過SAINT及project heads $g_1(.)$ and $g_2(.)$，是由MLP(one hidden layer and a ReLU)所組成的，在計算contractive loss之前可以減少維度，的確是可以改善訓練結果。
#### Loss functions 
在loss部分可以拆分成兩塊，如下：
1. Contrastive loss : 主要是想要相同資料點($z_i$和$z^{'}_i$)越靠近越好，反之鼓勵不同資料($z_i$ and $z_j$, $i \not = j$)點越遠越好，故這部分我們採用InfoNCE
2. Denoising Loss : 我們試著從雜訊的資料預測原始的資料樣本，假設我們給定 $r^{'}_i$及重構$x_i^{''}$，我們期望可以最小化原始與重構間的差異

The combined pre-training loss is 
$$\begin{equation} L_{pre-training}=-\sum^m_{i=1}log\frac{exp(z_i \cdot z_i^{'}/\tau)}{\sum^m_{k=1}exp(z_i \cdot z_k^{'}/\tau)}+\lambda_{pt}\sum^m_{i=1}\sum^n_{j=1}[L_i(MLP_j(r^{'}_i, x_i))] \end{equation}$$
where $r_i=S(p_i)$,  $r_i^{'}=S(p_i^{'})$, $z_i=g_1(r_i)$, $z_i^{'}=g_2(r_i^{'})$. $L_j$ is cross-entropy loss or mean squared error depending on the $j^{th}$ feature being categorical or continuous. Each $MLP_j$ is single hidden layer perceptron with a ReLU non-linearity. There are $n$ in number. one for each input feature. $\lambda_{pt}$ is a hyper-parameter and $\tau$ is temperature parameter and both of these are tuned using validation data. 
### finetuning 
透過SAINT進行無標註的資料的預訓練，接著可進行$l$個有標註資料的finetune。整理來說，對於特定$x_i$，可以學習相對應的contextual embedding $r_i$，在最後預測步驟，我們可以透過 embedding得到[CLS] token，接著透過簡單MLP得到最後的產出，我們就可以透過cross-entropy來衡量類別任務;透過mean squared error來衡量回歸任務