---
tags : Time Series, Deep Learning
---
NBeats
===
時間序列一直是商業上重要命題，過去都是以統計方式來解決，像是在M4 competition的冠軍是採用「neural residual/attention dilated LSTM stack」與「classical Holt-Winters statistical model」做結合，故作者想說是否可以存粹以DL進行模型預測，然而作者提出「backward」和「forward」的殘差進行連結，且此方式也具有解釋程度，並針對之前冠軍方式，提升11%，想當在時間序列領域上有的一大的進步。

## NBeats
Nbeats的全名為「Neural Basis Expansion Analysis For Interpretable Time Series Forecasting」，有兩個個屬性 1.基本架構為簡單且通用，並非複雜。2.此架構不依賴傳統的時間序列工程，最後此架構也能同步探索解釋程度。
