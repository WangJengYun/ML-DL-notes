---
tags: Machine Learning
---
Background
===
## 前言(Introduction)
Gradient Boosting Machine 目前在結構資料上最常使用的演算法之一，在 Kaggle 上所聽到的常勝軍 XGB、LightGBM 與 Catboost 都是基於這個概念進行調整與創新，本章節針對Gradient Boosting Machine基本架構進行說明，若有敘述不正確請不吝指教。
## 概念(Concept)
接著我們進一步了解Gradient Boosting，稱為梯度提升，是為弱學習模型的機器學習方法，它包含兩種概念，分別為梯度下降(grandient descent)與增強學習(boosting learning)，如下:
### 增強學習(Boosting Learning)
Boosting 的概念是相當簡單，主要透過一群弱學習器(weak Learner)做疊加，會根據前一個弱學習器(weak learner)所預測錯誤的sample在進行訓練，而獲得一個強學習器(strong Learner)可有較高準確率，這個方法主要是用於分類任務，其中最具有代表性的演算法為 Adaboost ，早期解決很多特定的困難分類問題。
#### Adaboost
Adaboost主要的概念如下圖，為每次迭代的弱學習器(week learner)，所將分類錯誤的樣本的權重放大，則下一次弱學習器抽到該筆樣本機率則會變大，其實就是錯誤中在學習。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Machine%20Learning/images/Background/Adaboost1.png?raw=true)

### 梯度下降(Grandient Descent)

## Gradient Boosting

