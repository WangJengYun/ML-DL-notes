---
tags : GNN
---
Knowledge Graph Reasoneing
===
Knowledge Graph Reasoneing 主要目標是在知識圖譜中從已存在的事實推測新的事實，KGR是可以進一步挖掘邏輯性規則，如   $(A, father of, B)+(A, hasband of, C) \rightarrow (C, mother of,B)$，這部分已變成發展快速的研究項目，在眾多AI應用項目中，逐漸證實KG有明顯的使用效益，像似：question answering 及 recommedation system，另外根據圖類別我們可以分類為三種(1) static KGR (2) temproal KGR (3) multi-modal KGR，此部分可以參考下方圖表示，早期研究 static KGR，其相關性質可分析直推性(transductive)及歸納性(inductive)，也有外推(extrapolation)與內推(interpolation)的性質