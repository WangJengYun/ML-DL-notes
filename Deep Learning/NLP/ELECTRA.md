---
tags : Tabular Data,Deep Learning
---
ELECTRA
===
過去在進行Masked lauguage modeling(MLM)的預訓練方法都使針對輸入的語句進行調整，像似BERT，使用[MASK]隨機替換原始tokens，並訓練模型進行重構，但這樣方式僅學習樣本的15%資訊，但是需要消耗龐大計算效能，故在本次所介紹ELECTRA方法，有效的解決，主要是採用「replaceed token detection」方法，主要是採用GAN概念，能有效學習文本語意，如下圖，故可以訓練較少的次數，就可以達到不錯的效果，另外若訓練較長時間，則可以