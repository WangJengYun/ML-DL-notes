---
tags : Tabular Data,Deep Learning
---
SAINT
===
過去Tabular Data已經有豐富的應用場景，像是詐欺預測、房價預測等等，此部分大多都是以gradient boosting與random forests演算法為主，處理分類及回歸問題，然而近期也會思考若以深度學習的方法會是如何應用？故本篇想與大家分享SAINT這個模型，主要是藉由2017年google所提出self-attention方法運用在columns與rows，切潛層layer會是以embedding作為轉換，作者有提到此方法是可以超越gradient boosting方法，像似XGBoost、CatBoost與LightGBM。
## Introduction
SAINT的全名為「self-Attention and Intersample Attention Transformer」，透過此方法可以克服一些訓練Tabular Data的困難。主要可將連續行變數及類別行變數映射到向量空間，並視為NLP的tokens當作transformer的輸入，接這transformer中有兩個方式來訓練，如下：
1. self-attention : 此部分主要關注每個資料點的個別的特徵
2. intersample attention：此部分可以藉由特定資料點於其他資料的關係來進行資料的分類，原理會是比較接近nearrest-neighbor classification。

最後此方法針對semi-supervised problem，採用self-supervied contrastive pre-training來增強訓練的結果。
## Self-Attention and Intersample Attention Transformer (SAINT)
Suppose $D=\{x_i,y_i\}^m_{i=1}$ is a tabular dataset with $m$ points, where each $x_i$ is an $n$-dimensional feature vector, and $y_i$ is a label or target variable. Similar to BERT, we append a [CLS] token with a learned embedding to each data sample. Let $x_i = [[CLS], f_i^{\{1\}}, f_i^{\{2\}}, f_i^{\{3\}},..., f_i^{\{n\}}]$ be a single data-point with categorical or continuous features $f_i^{\{j\}}$, and Let $E$ be embedding layer that embeds each feature into a $d$-dimensional space. Note that $E$ use different embedding functions for different features. For a given $x_i \in R^{(n+1)}$, we get $E(x_i) \in R^{(n+1 \times d)}$.
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/Tabular_Data/SAINT/SAINT.png?raw=true)

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
### Intersmaple attention
