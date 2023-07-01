---
tags : NLP,Deep Learning
---
RoBERTa
===
全名為Robustly optimized BERT approach，自從Bert發布之後，近年逐步推出基於transformer基礎的語言模型，其中包含ELMO、Bert、GPT跟XLNet，都是達到不錯的成績，而且不同任務的leaderborad也是前幾名，然而再2019下旬時Facebook提出RoBERTa的語言模型，主要Bert的改良版，調整的部分沒有很困難，其實是簡單粗暴的強力優化BERT的架構，另外再論文中也點出Bert其實是訓練不足(undertrained)，根據這樣的缺陷再做加強。
## 優化的項目
作者基於Bert架構做不同的實驗，找到可以改善的地方進而優化BERT訓練過程，調整面向如下:
1. 對於超參數所造成的影響有更嚴謹的研究
2. 提高訓練時間(Longer Training Time)
3. 增大訓練批次(Bigger Batches)
4. 使用更長的序列做訓練(Training Longer Sequence)
5. 使用動態產生MLM所使用的遮罩(Mask)
6. 使用更多更豐富的訓練資料(CC-News)

接著作者依照google所釋出的Bert架構$BERT_{BASE}(L=12,H=768,A=12,110M\  params)$為基礎進各種不同的實驗，所帶來的精準度的提升。
## Static vs. Dynamic Masking
原本的Bert是採用靜態的遮罩，也就是在資料預處理時就設定遮罩，所以在整個訓練過程中，都是使用相同的遮罩位置，然而在RoBERTa為了避免每筆資料在訓練過程中都是用同樣的遮罩，所以每筆資料使用10種不同的遮罩，所以RoBert總共會訓練40epoches，因此每筆資料會有四次使用到相同遮罩，這就是動態遮罩的邏輯。
動態遮罩是在每一次訓練都會提前生成遮罩模式(masking pattern)，在輸入到模型中進行訓練。整個實驗來看(下表)，兩種遮罩方式效果差不多，而動態遮罩大致上比靜態遮罩稍微好一點。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/RoBerta/Dynamic_Masking.PNG?raw=true)
## Model Input Format and Next Sentence Pediction
對於NSP(Next Sentence Prediction)存在不同的說法，像是Delin et al.(2019)表示移除NSP會讓模型的效度變低，但也有一些人(Lample and Conneau, Yang et al., Joshi et al., 2019)對NSP提出質疑，所以作者採取四種不同的做法來做實驗：
1. SEGMENT-PAIR+NSP:
  此設定與Bert相同，採取一對Segment可以包含多個句子，但整個長度不能大於512 tokens且採用NSP方式。
2. SENTENCE-PAIR+NSP:
   採取一對句子，因為句子通常會小於512 tokens，所以會增加Batch size，讓全部的tokens接近SEGMENT-PAIR+NSP方法的tokens數，另外也採用NSP方式
3. FULL-SENTENCES:
   在單一個或多個文本抽取句子，當作序列作為模型的輸入，是採用跨文本的句子，並插部特殊token來表示跨文本，另外則不採用NSP方式。
4. DOC-SENTENCES:
   主要僅在同一文本抽樣句子，但在靠近文本的結尾的地方，輸入的長度通常會小於512 tokens，此時則會動態提升Batch size方法，讓整理的tokens數靠近FULL-SENTENCES方法的tokens數，另外也不採用NSP方式。
   
從下表的實驗結果來看，可以發現我們個別使用句子，是會傷害下游任務的效度，作者猜測是因為沒有讓模型學習到距離較長相依性(long-range dependencies)，另外也可以得知移除NSP會稍微改善下游任務的效度，最後使用DOC-SENTENCES稍微比FULL-SENTENCES較好，發現跨文本其實會影響模型效度，但DOC-SENTENCES會使batch-size變得不固定，所以最後RoBERTa採用FULL-SENTENCES進行後續的實驗。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/NLP/RoBerta/Next_Sentence_Pediction.PNG?raw=true)