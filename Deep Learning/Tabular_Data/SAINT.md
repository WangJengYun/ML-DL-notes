---
tags : Tabular Data,Deep Learning
---
SAINT
===
過去Tabular Data已經有豐富的應用場景，像是詐欺預測、房價預測等等，此部分大多都是以gradient boosting與random forests演算法為主，處理分類及回歸問題，然而近期也會思考若以深度學習的方法會是如何應用？故本篇想與大家分享SAINT這個模型，主要是藉由2017年google所提出self-attention方法運用在columns與rows，切潛層layer會是以embedding作為轉換，作者有提到此方法是可以超越gradient boosting方法，像似XGBoost、CatBoost與LightGBM．
## Introduction
