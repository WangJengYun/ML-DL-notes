---
tags : NLP,Deep Learning
---
Bert
===
## 前景提要
今天跟大家介紹NLP的里程碑Bert(Bidirectional Encoder Representations from Transformers)，此模型於2018年底由Google推出，造成轟動，推出不久就已經在問答資料庫SQuAD 2.0超越人類的問答分數(如下)，是相當具有代表性的模型。
![image alt](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/Bert/Bert1.png?raw=true)

論文提到，此模型是基於左右上下文進行訓練，也是有說再針對句子理解時，句子的每一個字詞都會根據上下文的詞意進行訓練，了解當下字詞的語意，另外此模型分為兩階段訓練為預訓練(pre-training)與微調(fine-truning)，且可以根據這個機制運用於多元的任務中，像是問答(question answering)、語言推論(language inference)與其他分類任務等等。
## Bert
如前面所說，在Bert模型中訓練分為兩個階段為預訓練(pre-training)與微調(fine-truning)(如下圖表示)。在預訓練過程中，是採用沒有標記的資料(unlabeled data)來訓練不同任務，另外對於微調(fine-truning)的部分，相關初始化參數是沿用預訓練的參數，再根據不同的任務進行有標記的資料(labeled data)訓練， 這樣的機制可以減少對龐大的資訊訓練所造成資訊成本，也就是說Google已經根據不同語言預前進行預訓練，做初步語意理解，然而在運用在商業問題，僅使用少量資料就可以得到不錯的結果，這就是遷移學習(Transfer Learning)，另外預訓練與微調的模型架構是相同的，
Google釋出兩規格的模型，如下:
* $Bert_{base}(L=12, H=786, A=12 Total Parameters=110M)$
* $Bert_{LARGE}(L=24, H=1024, A=16 Total Parameters=340M)$

where $L$ is the number of layers(i.e. Transformer blocks),$H$ is the hidden size, $A$ is the number of self-attention heads.
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/Bert/Bert2.png?raw=true)
## Pre-traininig Bert 
Bert的機制中，以非監督式學習方式對未標籤資料進行訓練語意，這裡不想以前的訓練方式，以單向式訓練(左到右或右到左)，Bert是採取雙向以非監督式訓練，另外Bert會兩種任務進行非監督式訓練，如下:
### Task #1 : Masked LM
我們想要了解字詞的語意，所以在這裡採取方式與Word embedding(word2vector、Fastext)方法相同，也就是在輸入一段句子，會有15%隨機遮蔽字詞，然後模型針對字詞為預測目標進行訓練，就是從上下文來推測這個字詞是什麼，輸入可能得類別為所有的詞彙並藉由模型給予相對的機率，另外論文也提出 pre-training 與 fine-tuning 會造成不匹配，因為[mask]這個token不會再fine-tuning出現，所以為了紓解這個議題，我們部會總是以[mask]來替換被遮蔽的字詞，所以我們會有15%機率選取字詞來當作預測目標，另外被選到字詞，會依造下列機率替換:
1. 80%機率以[mask]替換
2. 10%機率以隨機任意的token替換
3. 10%機率以不做任何的替換

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/Bert/Bert3.png?raw=true)
### Task #1 : Next Sentence Prediction (NSP)
很多重要的NLP任務像是問答(Question Answering)或者自然語言推論(Natural Language Inference)都是需要基於兩個句子之間的相關性，所以Bert為了可以抓到這個相關性，採取二元分類任務，輸入兩斷句字，且已[SPE]的token做為區隔，來預測是否為下一句，另外在Input的[CLS]的token會將輸入的資訊進行壓縮或聚集，讓模型有足夠的資訊預測句子個關聯性。在二元分類任務中，需要負樣本來協助模型學習我們的目的，所以訓練樣本為下列:
1. 50%的占比句子A會實際下一句為句子B，標註為IsNext
2. 50%的占比句子A會接這隨機挑選句子B，標註為NotNext

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/Bert/Bert4.png?raw=true)

## Fine-tuning Bert
Bert在執行下游的預測任務時，會採用預訓練好的模型參數，這樣就可以已少量的資料就可以達到不錯的預測精準度，而且我們採與句子A與句子B進行訓練，之後再Fine-tuning時，可以接受不同的pair任務，如下:
1. 改敘或換句話說的兩個句子
2. 從推論去辨認的蘊含關係
3. 問答關係
4. 文本分類(Text-X)或者句子標記(命名實體辨認)

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/Bert/Bert5.png?raw=true)
#### Bert for feature extraction 
Fine-tuning不只是運用在各個場景上，也可以建立各個情境的word embeeding，我們以命名實體辨認舉例，我們該用哪一層的word embeeding進行預測呢?我們一下張圖來看，是做不同的實驗將不同層的word embeeding進行聚集(平均或加總等等)，且根據我們目的評估F1 score，你會發現用整個模型所的到96.4，而運用不同方的聚集相對較低，但可以發現哪一方式會比較靠近96.4，所富含資訊相對豐富。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/Bert/Bert6.png?raw=true)
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/Bert/Bert7.png?raw=true)