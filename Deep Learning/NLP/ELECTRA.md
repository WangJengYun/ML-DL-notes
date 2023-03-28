---
tags : Tabular Data,Deep Learning
---
ELECTRA
===
過去在進行Masked lauguage modeling(MLM)的預訓練方法都使針對輸入的語句進行調整，像似BERT，使用[MASK]隨機替換原始tokens，並訓練模型進行重構，但這樣方式僅學習樣本的15%資訊，但是需要消耗龐大計算效能，故在本次所介紹ELECTRA方法，有效的解決，主要是採用「replaceed token detection」方法，主要是採用GAN概念，能有效學習文本語意，如下圖，故可以訓練較少的次數，就可以達到不錯的效果，另外若訓練較長時間，則可以達到當時候SOTA模型。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ELECTRA/ELECTRA_1.jpeg?raw=true)

## Introduction 
過去在NLP的訓練模型通常會分成兩類如下，左邊為lauguage modeling(LM)，以GPT作為主要代表，此類的模型是依照從左到右的順序輸入文本並進行處理，依據前文來預測下一次詞，其缺點是只是僅用當前的詞進行預測，所參考的資訊較少;第二個為Masked lauguage modeling(MLM)，以BERT作為主要代表，這類性的模型，會先將預輸入的文本以隨機方式進行遮蔽(如BERT使採用15%機率)，並預測被遮蔽的詞，此類是具有雙向性的，但也是有缺點，就是沒有預測每一個token，而是預測很小的子集(每一批次的15%)，可能會減少沒個語句的訊息量，故本次論文提出「ELECTRA」的方法進行改善。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ELECTRA/ELECTRA_2.gif?raw=true)

ELECTRA使參考生成對抗網絡(Generative Adversarial Network, GAN)提出「replaceed token detection」的機制，其架構如下。主要是讓模型可以區分「真實」與「虛假」的token。先將特定token以假的token替換，接著透過模型來進行訓練並判定哪些已經被替換的token還是與原來一樣的，使用這個方法你會發現每個token的資訊都會學習到，而且僅用少量的替換就可以實現相同的表現，甚至在下游任務表現比BERT還要好，而這個機制會比較想GAN訓練的概念，接下來再跟大家說明一下模型細節。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ELECTRA/ELECTRA_3.gif?raw=true)

## ELECTRA
基於上述的概念，ELECTRA會有兩個網絡模型(如下)，一個是生成器(generator) $G$ ，另一個為判別器(discriminator) $D$，這兩格網絡模型都是由一個編碼器(encoder)所組成(如：Transformer network)，將輸入的token映射到語境向量(contextualized vector)，其個別任務會有所不同，我們會藉由生成器來隨機生成假的token替換，讓判別器來學習哪一個token是假的，這樣你會跟過往會不一樣，不只是學習被遮罩的token，而是全部的token都會讓模型學習，故下游的相關任務表現都會比較好。
