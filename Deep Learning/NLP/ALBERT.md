---
tags : NLP,Deep Learning
---
ALBERT
===
近期語言模型發展中，為了提高下游任務表現都是增加模型大小，然而所使用的內存資源(CPU/GPU)相對也越來越大而且還有模型衰退現象(Model Degradation)，為了解決這個問題，google於2020年提出ALBERT，其中包含有幾種方式或概念來減少模型的參數，並增加模型訓練的速度，也與原始BERT做比較，在各個面向都是優於原始BERT，接著一一與大家介紹ALBERT做了什麼事情?
## 優化的項目
ALBERT核心目標希望可以減少訓練的參數量並減少訓練時間，藉此提供兩個參數縮減的技術及一個改進的方式提升下游任務的表現(如下)，但是模型的層數不會減少，故模型的推理時間(Inference Time)沒有藉由此模型而改進，不過整理來說，與BERT相比已經大幅減少訓練的時間及精準度的提升。
1. Parameter Reduction : factorized enmbedding parameterization
2. Parameter Reduction : Cross-layer parameter sharing
3. Improvement : Inter-sentence coherence loss

## Factorized Enmbedding Parameterization
在BERT及衍生的模型(XLNet、RoBERTa等等)中，所使用的WordPiece的Embedding維度與隱藏層(Hidden Layer)的大小對齊，這樣才能投射到隱藏空間，那這樣其實會產生較多的參數需要訓練，假設共有30K的詞彙需要投射到768維度隱藏層，則需23040000參數值需要訓練。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ALBERT/Factorized_Enmbedding_Parameterization_V1.PNG?raw=true)
這個做法會以兩個方式切入，說明其限制:
1. 模型角度:
WordPiece embedding主要學習是不依賴上下文內容(context-independent); Hidden Layer embedding的學習是依賴上下文內容(context-dependent)，這兩個可以獨立拆開，也就是把E跟H拆開，可以更高效的是使用模型所需的參數，則結果為$H>>E$
2. 實踐的角度:
通常語言模型會需要大小為$V$的詞彙做訓練，如果$E \equiv H$則會增加embedding維度，其大小為$V \times E$，但實際訓練也會比較稀疏。

綜合以上考量，ALBERT提出矩陣分解，將WordPiece embedding拆解為兩個小矩陣，分別為$V \times E$和$E \times H$(如下)，先將投射到大小為$E$的空間，在投射到隱藏空間(Hidden space)，整個參數量為由$O(V \times H)$到$O(V \times E + E \times V)$，其結果很明顯$H>>E$
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ALBERT/Factorized_Enmbedding_Parameterization_V2.PNG?raw=true)
從下列實驗來看，也發現E的大小與模型精準度並無正向關係，則後續作者採用$E=128$
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ALBERT/Factorized_Enmbedding_Parameterization_V3.PNG?raw=true)
## Cross-layer Parameter Sharing
ALBERT也提供另一個方式減少參數量，就是跨層共享參數(Cross-layer Parameter Sharing)，這樣可以有效利用參數進行訓練，另外也提供不同的共享方式如(1) 僅Feed-forward network parameters (2) attention parameter (3) all parameter。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ALBERT/Cross-layer_Parameter_Sharing_V1.PNG?raw=true)
另外做作者藉由這個方式來觀察每一層input與output的L2 distance及cosine similarity，來觀察是否動盪而不是區收斂於0，由下圖可已知道ALBERT相對比BERT來的更平滑，這樣表示有穩定模型參數的效果，雖然一個下墜點但是經過24層沒有收斂到0，都還算可接受範圍內。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ALBERT/Cross-layer_Parameter_Sharing_V4.PNG?raw=true)
整理的參數量設計如下，可以發現採用跨層共享參數可大幅減少參數量。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ALBERT/Cross-layer_Parameter_Sharing_V3.PNG?raw=true)
由下表發現讓精準度下降主要因為共享Feed-forward network parameters，反之attention parameter不太會影像精準度，但雖然整體來說採用採用Factorized Enmbedding Parameterization跟Cross-layer Parameter Sharing會讓精準度下降，但也能達到參數大幅度的縮減。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ALBERT/Cross-layer_Parameter_Sharing_V2.PNG?raw=true)
## Inter-sentence Coherence Loss
說明Inter-sentence Coherence Loss方法之前，先來回顧BERT的Next-Sentence Prediction，這個是二分類法來預測這兩個片段文本是否連續的，就是為上下文，在訓練過程中，會先準備正/負樣本提供訓練，正樣本就是重同一篇文章取上下片段的文本，負樣本為不同的文章中取出片段的文本，示意圖如下，主要這個方式要來改善下游任務的表現，但是在後續的paper點出NSP的缺點，如RoBERTa的paper指出NSP對於下游任務的影響不大，則將其移除，然而作者認為NSP訓練困難度不夠，容易搞混主題預測(Topic Predition)和語句的連貫預測(Coherence Prediction)搞混，因為主題預測相對比連貫預測的學習更容易，另外與MLM loss重疊滿多。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ALBERT/Inter-sentence_Coherence_Loss_V1.PNG?raw=true)
ALBERT這是還是保留預測連貫性的機制，因為還是想要模型學習語意，然而這裡是採用Inter-sentence Coherence Loss來學習，主要是避免主題學習，而完全專注學習語句的連貫性，所以這裡所使用的正負樣本會不太一樣，正樣本也時從相同文章中取出上下片段文本，負樣來則是這兩個片段文本交換，來藉此學習上下文的順序，可以更細緻判別話語連貫性，示意圖如下。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ALBERT/Inter-sentence_Coherence_Loss_V2.PNG?raw=true)
由下表了解，SOP可以解決NSP無法解決的議題，且在多語句編碼任務下可以很直接提升下游人任務的表現。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ALBERT/Inter-sentence_Coherence_Loss_V3.PNG?raw=true)

## Summary
最後作者提到ALBERT-xxlarge達到比BERT-large更好的效果，但過長的訓練時間跟推論時間(inference time)還是需要再被改進，下一步會在思考Sparse attention與block attention來改善這些缺點，另外關於SOP，作者認為當前的方法還未抓到更多維度資訊，可能還會有好的Loss設計來改善。