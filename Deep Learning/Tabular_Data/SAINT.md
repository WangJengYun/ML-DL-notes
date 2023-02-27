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
