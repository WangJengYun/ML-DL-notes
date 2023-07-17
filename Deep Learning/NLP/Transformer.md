---
tags : NLP,Deep Learning
---
Transformer
===
## 前景提要
近期，在NLP技術發展階段，經過傳統的機率模型至RNN與CNN，但是都有缺點存在，然而Google於2017提出Transformer，取代過去的RNN技術，可能萃取更多文本資訊並且提升計算效率，但是模型的本體基於過去的研究提出自注意機制的Seq2Seq模型，其結構也是由Encode與Decode所組成的。
![](https://i.imgur.com/uwhlcEV.png)
### Encode-Decode model
在前幾篇文章，已經介紹Encode-Decode的機制，當時Encode與Decode在Seq2Seq僅用一層，在Transformer則疊加很多層為Encode component，且Encode的資訊傳遞至對應相同層數的Decode，如下列的示意圖，這樣的方式可以萃取更多資訊，讓模型學習到更多語意及上下文資訊。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/transformer/encode_decode_model.png?raw=true)
## Overview of Model Architecture
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/transformer/transformer_model.png?raw=true)
### Encode-Decode model
Transformer基本架構也是Encode-Decode Model，其中Encode部分是將輸入序列$x = (x_1,x_2,x_3,...,x_n)$對應到連續序列$z= (z_1,z_2,z_3,...,z_n)$，另外Decode的部分，是給定$z$，在每個預測的當下生成輸出序列$(y_1,y_2,y_3,...,y_m)$，這部分為自回歸機制(auto-regression)，在每一次生成下一個字詞時，都會基於前一個生成的字詞作為額外的輸入。
#### Encode
Encode部分可以由上圖得知，會是由N個層堆疊起來$(N=6)$，其中每一層是由兩個子層組合而成，第一子層為Multi-Head Self-Attention機制，第二個子層為pointwise fully connection feed-forward network，另外採用殘差連結(Residual Connection)與layer normalization運用於每個子層，也就是每個子輸出為$LayerNorm(x+Sublayer(x))$，其中Sublayer可以為上述的兩個函式，這樣可以讓模型加深學習，實驗過程中維度$d_{model}$可以達到512。
#### Decode
Decode部分也是由N層堆疊起來$(N=6)$，但是子層鑲入三層，在第一個子層，有特別修改，為了避免當下的輸出字詞，含有未來的資訊，所以採用遮蔽的機制(masking)，確保預測當下的第$i$字詞是基於前幾個字詞有關，其位置會小於$i$，其餘的部分與encode很類似，也採用殘差連結(Residual Connection)與layer normalization運用於每個子層，加深模型的訓練。
### Attention
Attention在NLP領域中是一個里程碑，突破過往的language model的限制，直到2017年google全面將模型的核心以Attention為主，帶來很不錯的效度。從字面來看，會是希望他讓模型可以訓練所關注的資訊，舉個例子來說，我們有一段句子如下:
$$The\ animal\ didn't\ cross\ the\ street\ because\  \textbf{it}\ was\ too\ tired$$我們會思考it會代表什麼(animal還是street)?，對我們來說這是很簡單的問題，但對演算法會是困難的，然而在attention可以很精認為與aniaml有關聯，這個結果突破過去技術，達到語意理解的程度。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/transformer/attention.png?raw=true)
接這進而探討transformer的attention機制，稱為self-attention，這是主要從input產生query、key與value三個向量，並將其映射於output，中間過程是對value進行加權加總而得到的，其中weight由query與key計算而來的。
#### Scaled Dot-Product Attention
接這我們看如何計算self-attention，整個過程是如何運行。下圖為示意圖
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/transformer/Scaled%20Dot-Product%20Attention.png?raw=true)
* **The Step 1** : 
   首先self-attention會從encoder的輸入向量(word embedding)生成三個向量為query、key與value，維度可由$d_k$、$d_k$與$d_v$表示，其中query與key是依樣的，另外由每個字的詞向量組成，堆疊成矩陣格式進行後續的計算，這個部分要注意的事，所生成向量的維度會比輸入向量小，論文表示生成向量為64;輸入向量為512。
*  **The Step 2** :     
  接下來我們要計算分數衡量當前字詞如何關注其他字詞，所以會以query與key向量進行內積(dot product)
*  **The Step 3** : 
  所計算的分數進行正規化，除於$\sqrt{d_k}$，這是避免梯度消失的議題，也就是說當$d_k$較大時，會的到較大內積，這樣會在softmax得到極小的梯度值，因此為了抵銷此效應，則除於$\sqrt{d_k}$進行正規化。
*  **The Step 4** :   
  所的數值進行softmax轉換，表示每個字詞的關注/重要程度，這是非常有用的，讓我們知道與當前的字詞的相關程度。
*  **The Step 5** : 
  最後經過softmax所得到的數值與value進行相乘並加總，這個就可以的到當前字詞的輸出向量。

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/transformer/Scaled%20Dot-Product%20Attention_single%20vector.png?raw=true)
#### Multi-Head Attention 
論文再進一步堆疊多個self-attention，稱為multi-head attention，這可以強化模型的表現，讓模型於不同 heads 在不同的 representation subspaces 裡學會關注不同位置的資訊，也就是會映射$h$不同的query、key與value到$d_k$、$d_k$與$d_v$的維度，再分別平行映射於$d_v$j維度的輸出向量，最後將這些串接，在一次映射得到最終的數值，其公式如下:
$$MultiHead(Q,K,V)=Concat(head_1,head_2,...,head_h)W^O$$
where $head_i = Attention(QW_i^Q,KW_i^k,KW_i^V)$ and the projections are parameter matrices $W_i^Q \in \mathbb{R}^{d_{model}\times d_k}$,$W_i^K \in \mathbb{R}^{d_{model}\times d_k}$,$W_i^V \in \mathbb{R}^{d_{model}\times d_v}$,$W_i^O \in \mathbb{R}^{d_{model}\times hd_v}$.

另外再使用Multi-Head Attention會有三個地方
1. 在encoder-decoder attention，qeury使從decoder層來，另外key和value使從encoder來，這樣可以讓decoder每個位置都可以參考到輸入的序列每一個位置。
2. 在encoder層，query、key與value都是從相同來源產生，在這個cased可以表示讓encoder每個位置都可以參考到encoder的前一層的每個位置。
3. 在decoder層，與上述的機制很相似，decoder每個位置都可以參考到decoder的每個位置，但是我們需要避免未來訊息，所以納入自回歸的機制，也就是在執行self-attention會去遮蔽不合理的連接所生成的數值(設定為-$\infty$)。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/transformer/multi-head%20attention.png?raw=true)
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/transformer/multi-head%20attention%20all%20.png?raw=true)
我們來看實際的case為什麼要使用mutl-head attention，如下列右圖表示，與it有關聯會有不同意涵，第一個it是誰的代名詞，所以你會發現左側的animal顏色比較深，另外it的狀態表示，會發現右側的tire顏色較深，這樣可以說明可以藉由mutl-head attention還表示不同與it的關聯，讓模型更能理解語意。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/transformer/multi-head%20attention_example%20.png?raw=true)
### Positional Encoding
由於transformer都以self-attention為基礎，並沒有遞迴或者捲積，但是為了讓模型可以抓到序列(sequence)，所以在納入序列的絕對位置在模型之中，也就是增加positional encoding於input embedding，且維度與input embedding相同為$d_model$，此方法實際是採用sine與cosine函式，如下:
$$PE_{(pos,2i)} = sin(pos/1000^{2i/d_{model}})$$$$PE_{(pos,2i+1)} = cos(pos/1000^{2i/d_{model}})$$
where $pos$ is the position and $i$ is dimension.
論文之所以選擇此方法，因為可以讓模型容易學習相關位置，


### Residuals