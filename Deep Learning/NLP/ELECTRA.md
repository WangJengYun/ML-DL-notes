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
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ELECTRA/ELECTRA_4.png?raw=true)

A generator $G$ and a discriminator $D$ primarily consists fo an encoder(e.g. a Transformer network) that maps a sequence on input tokens $x = [x_1, ..., x_n]$ into a sequence of contextualized vector representations $h(x) = [h_1,...,h_n]$. For a given position $t$(in our case only positions where $x_t = [MASK]$), the generator outputs a probability for generating a particular token $x_t$ with softmax layer:
$$p_G(x_t|x) = \frac{exp(e(x_t)^Th_G(x)_t)}{\sum_{x^{'}}exp(e(x^{'})h_G(x)_t)}$$
where $e$ denotes token embeddings. For a given position $t$, the discriminator predicts whether the token $x_t$ is 「real」 i.e., that is comes from the data rather than the generator distribution with a sigmoid output layer :   
$$D(x,t) = sigmoid(w^Th_D(x)_t)$$
The generator is trained to perform masked laguage modeling(MLM). Given an input $x = [x_1, x_2, ..., x_n]$, MLM first select a random set of postions(integers between 1 and n) to mask out $m = [m_1, ... , m_k]$. The tokens in the selected positions are replaced with a $[MASK]$ token: we denote this as $x^{masked} = REPLACE(x,m, [MASK])$. Formally model inputs are constructed according to 
$$m_i ～ unif\{1,n\}\ for\ i\ =\ 1\ to\ k$$
$$x^{masked}=REPLACE(x,m,[MASK])$$
$$\hat{x}_i ～ p_G(x_i|x^{masked})\ for\ i \in m$$
$$x^{corrupt} = REPLACE(x,m,\hat{x})$$
and the loss functions are 
$$L_{MLM}(x,\theta_G) = E(\sum_{i \in m}-log p_G(x_i|x^{masked}))$$
$$L_{Disc}(x|\theta_{D}) = E(\sum^n_{t=1}-\Bbb{1}(x_t^{corrupt}=x_t)logD(x^{corrupt},t)-\Bbb{1}(x_t^{corrupt}\neq x_t)log(1-D(x^{corrupt},t)))$$
由上述的公式可以知道，我們可以透過生成器學習預測被遮罩的token，然而判別器學習生成器所生成的token，依照這樣的邏輯，我們就可以讓判別器是識別生成器所生成的token是否正確/是否與原始的token一樣，讓模型都可以學習到每個token資訊。

你會發現此概念類似GAN的訓練機制，但還是有些微不同，例如：如果生成器預測正確的token，此token還是會被視為「real」而不是「fake」，作者發現依照這樣機制是可以有效地改善下游任務的表現，更重要的事，在生成器訓練時是採用「maximum likelihood」當作loss function，而不是跟GAN一樣對抗式愚弄判別器，另外作者也由提到我們比較難以透過生成器所採生的樣本進行對抗式方式訓練，但也有做過相關的實驗，像似以強化學習(reinforcement learning)對生成器進行訓練，表現還是比較差，故不會支持這個做法。最終的loss function 如下：
We minimize the combined loss 
$$min_{\theta_G, \theta_D} \sum_{x \in \chi}L_{MLM}(x, \theta_G) +\lambda L_{Disc}(x, \theta_D)$$
over a large corpus $\chi$ of raw text.
以上我們是不會透過生成器來針對判別器進行反向傳播，且在預訓練後，會將提供判別器予下游任務進行使用。

## ELECTRA的相關實驗
### Weight Sharing 
作者思考如果將參數共享會不會提升精準度，故若將生成器及判別式的的weight拉齊一至，則必須將生成器及判別式的size設為一樣，然而在實驗過程中發現若採用小型的生成器，其實效果更好，但這類作法僅分享embedding層(token與position embedding)，且採用embedding的長度設置跟判別式的hidden states一樣．我們比較各個做法的效果(1) 生成器的大小與判別器的大小一至，則訓練500步且沒有固定權重，其GLUE分數為83.6; (2) 僅固定embedding權重，則GLUE的分數為84.3; (3) 固定所有的權重，則GLUE的分數為84.4．這樣的實驗結果，發現固定token權重相對效果明顯，因為生成器的embedding有更好的學習能力，是因為透過生成器的softmax更新所得詞彙，然而在判別器僅會更新已生成器所抽樣的token或者已存在的token，最後僅會固定embedding層進行訓練．
### Smaller Generators
作者還有進行模型的大小做實驗，下圖的左邊可以發現，採用1/4-1/2的判別式大小的生成器表現較好，作者推測使用太強壯的生成會使在判別器的訓練更加困難，
而且會主要使判別器必須要針對生成器的結果進行建模而不是真實數據分佈，故之後會採用較小的生成器．
### Training Algorithms
最後作者採用不同的模型架構來進行訓練，包含上述所講的原始ELECTRA的架構，共有三個，其中第一個與第二個如下：

(1) Two-Stage ELECTRA：首先先訓練n步驟的生成器$L_{MLM}，並採用生成器的權重將判別器$L_{Disc}$的權重初始化，再訓練n步的判別器$L_{Disc}$且固定生成器的權重。

(2) Adversarial ELECTRA:此部分為了解決判別器的反向傳導無法生成器，故採用reinforcement learning解決這個問題．

(3) ELECTRA:相關模型架構，可參考上一小節的說明．

我們可以由下圖發現，可見原始ELECTRA的結果是最好的，其次是Adversarial ELECTRA，但是還是無法超越原始ELECTRA，其原因作者推測試強化學習的較差的樣本效率及在生成器產生low-entropy輸出分配，其大部分的機率是在單一個token上，則在生成器生成的樣本較缺乏多樣性，另外對於Two-Stage ELECTRA隨然比較差但還是比bert表現較好。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/ELECTRA/ELECTRA_5.png?raw=true)