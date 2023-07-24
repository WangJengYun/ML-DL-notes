---
tags : GNN
---
Knowledge Graph Reasoneing
===
Knowledge Graph Reasoneing 主要目標是在知識圖譜中從已存在的事實推測新的事實，KGR是可以進一步挖掘邏輯性規則，如   $(A, father of, B)+(A, hasband of, C) \rightarrow (C, mother of,B)$，這部分已變成發展快速的研究項目，在眾多AI應用項目中，逐漸證實KG有明顯的使用效益，像似：question answering 及 recommedation system，另外根據圖類別我們可以分類為三種(1) static KGR (2) temproal KGR (3) multi-modal KGR，此部分可以參考下方圖表示，早期研究 static KGR，其相關性質可分析直推性(transductive)及歸納性(inductive)，也有外推(extrapolation)與內推(interpolation)的性質。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/Deep%20Learning/image/GNN/Knowledge%20Graph%20Reasoneing/Knowledge%20Graph%20Reasoneing_1.png?raw=true)
## Basic Definition
在定義不同類型的知識圖譜之前，請參考下方數學符號表示：

### Static Knowledge Graph
首先我們介紹常見的靜態知識圖譜，此類型通常紀錄在橫斷面的多元節點及關係，相似客戶A購買產品B之類的，不同的head to tail 所組成的知識圖譜，也是常常使用此結構來紀錄各領域的知識點，相關定義可以參考下列數學說明/公式：
#### Definition 
Static knowledge graph is defined $SKG=\{E,R,F\}$, where $E$, $R$ and $F$ represent the sets of entities, relations, and facts. The fact is in a triplet format $(e_h,r,e_t) \in F $, where $e_h,e_t \in E$ and $r \in R$ between them. Note that the static knowledge graph is known as the conventional knowlege graph. The phrase "static" is used to distinguish it from other types of knowledge graphs.
#### Reasoning 
Give a static knowledge graph $SKG=\{E,R,F\}$, knowledge graph reasoning aims to exploit the existing facts to infer queried fact $(e_h^q,r_q,e_t^q)$. Three sub-tasks are defined, i.e head resoning $(?,r_q,e_t^q)$, tail inferring $(e^q_h,r_q,?)$ and relation inferring.

### Temporal Knowledge Graph
#### Definition 
Temporal knowledge graph is defined as a sequence of static knowledge graphs in different timestamps $TKG=\{SKG_1,SKG_2,...,SKG_t\}$. The $KG$ at timestamps $t$ is defined as $SKG_t = {E,R,F_t}$, where $E,R$ are the sets of entities and relations, $F_t$ is the set of facts at timestamp $t \in T$. The quadruple fact $(e_h,r,e_t,t)$ represents that relation $r$ exists between head $e_h$ and tail $e_t$ at timestamp $t$
#### Reasoning 
Give a temporal knowledge graph $TKG=\{SKG_1,SKG_2,...,SKG_t\}$, where $SKG_t = \{E,R,F_t\}$ and timestamp $t \in T$. With the queried fact 
$(e_h^q,r_q,e_t^q,t_q)$
 reasoning can be classified into three types, including entity reasoning i.e $(?,r_q,e_t^q,t_q)$ and $(e_h^q,r_q,?,t_q)$, relation reasoning $(e_h^q,?,e_t^q,t_q)$ and timestamp reasoning $(e_h^q,r_q,e_t^q,?)$ The former two sub-task are similar to the reasoning on static and multi-modal $KGs$, while the latter one is unique in temporal $KRG$

### Multi-Modal Knowledge Graph
#### Definition
Multi-modal knowledge graph $MKG$ is composed of knowledge facts where more than one modalities exist. As an early-stage research field, the relevant definitions are not systematic enough. Generally speaking, according to the representation mode of other modal data, there are two multi-modal $KG$ i.e $N-MMKG$ and $A-MMKG$ Moreover, we also count multi-modal temporal $KG$ as one type of $MKG$ though there are few works.
#### Reasoning
Reasoning over multi-modal knowledge graph reasoning is similar to the reasoning task for the other two $KG$ types i.e inferring the missing facts either in triple or quadruple format. But since the entities are in more than one modalities in multi-modal $KGs$, multi-modal $KGR$ generally requires extra knowledge fusion modules for different modalities before fact inference.
