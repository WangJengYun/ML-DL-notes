---
tags : GNN
---
Graph Convolutional Network(GCN)
===
在說明GCN之前我們先來談談傅立葉轉換(Fourier Transform)，傅立葉轉換主要是將由時域轉到頻域的技巧，且通過一個濾波器後，再透過傅立葉反轉換，將頻域訊號轉回時域，透過轉換及反轉可以協助分解不同頻譜並解析，目前可以運用到不同的領域上，如下：
1. 把整首歌曲分成成千上萬個小部分，再從中截去不重要的頻率，留下重要的頻率，最後被耳朵聽到的音軌裡只剩下那些被留下來的最重要的頻率（音調）。這樣處理過後，檔案容量通常只要原本的十分之一，而我們也很難分辨壓縮後的音頻和原始音頻的區別。
2. 主動降噪耳機同樣也使用傅立葉轉換：耳機裡內建的麥克風可記錄你周圍環境的噪音，並在所有頻譜裡測量噪音的頻率，之後這些噪音的頻率會被反轉，然後混合到你的聲音內容裡頭，這樣能抵消你身邊嬰兒的哭聲或公路上的噪音。降噪的原理是「反相」而非「反向」，利用波在週期內部的對稱性，製造出一個振幅相同，但相位相反的波動，實現相互的抵消。聲波本質是機械振動，而且人耳的聽力範圍也有限（20Hz~20KHz），在此頻率範圍中，無論透過耳機還是音響，產生降噪所需的波動，進而讓波動相消。

初步了解傅立葉轉換，基於頻譜的GCN不在是時域資訊，而是圖訊號(如圖一)，透過圖訊號轉出，可以了解每個節點所到資訊，也是與時域的訊號相同(如圖二)，若是低頻表示固定距離內的節點，訊號差距較小，高頻則反之。簡單來說，頻率越高與鄰居節點的訊號差距愈大(如圖三)。

## 基礎圖概念
A graph can be defined as $G = (V,E,A)$, where $V$ is the set of vertices or nodes, $E$ is set of edges or links, and $A$ is the adjacency matrix of size $n \times n$, $v_{i} \in V$ denotes a node, and $e_{ij} \in E$ denotes an edge connecting $v_i$ and $v_j$ in a graph $G$. If an edge $e_{ij}$ in graph, denoted by $e_{ij} \in E$, then $A_{ij}>0$, otherwise $A_{i,j}=0$ and $e_{i,j} \notin E$

### Degree of vertex
The degree of node $i$ is $d_{i}$, representing the number of edges connected to node $i$, which is defined by 
$$d_i = \sum^n_{j=1} 1_{E}\{e_{ij}\}$$
where $1$ is indicator function.
Given a grah $G$, the degrees matrix $D \in \mathbb{R}^{n \times n}$ is 
$$D_{i,j} = \begin{cases} d_i  & \text {if i=j} \\ 0 & \text{otherwise} \end{cases}$$
Undirected graph is graph with undirected edges and has $A_{i,j} = A_{j,i}$. In contrast, directed Graph is graph with directed edges, which may not satisfy $A_{i,j} \not=	 A_{j,i}$. The spectial graph convolutional network is defined on an undirected graph. In fact, an undirected graph is a special case of a directed graph.

## 拉普拉斯矩陣(Laplacian matrix)
有了基本概念，接下來說明拉普拉斯矩陣，此數學理論為譜圖理論中的核心與基本概念，在機器學習及深度學習有蠻重要的應用，尤其是圖學技術方面。

Weight on the graph is associated numerical value assigned to each edge of a graph. A weighted graph is graph associated with a weight to each of its edges while an unweighted graph is one without weights on its edge. The Laplacian maxtrix(unnormalized Laplacian or combinatorial Laplacian) for an unweighted graph is 
$$L=D-A \in \mathbb{R}^{n \times n}$$
Analogously, weighted graph is 
$$L=D-W \in \mathbb{R}^{n \times n}$$
where $W$ is weighted adjacent matrix
### Eigen Decompostion
it also known as spectral decompostion, is a method to decompose a matrix into a product of matrics involving its eigenvalues and eigenvectors. Assuming base of $\mathbb{L}$ is $U = (\overline{u}_1,\overline{u}_2,...,\overline{u}_n),\overline{u}_i \in \mathbb{R}, i = 1,2,...,n$.Considering the Laplacian matrix is real symmetric, the spectral matrix, the spactral decompostion of Laplacian matrix is 
$$L = U \Lambda U^{-1} = U \Lambda U^{T}$$
where 
$$\Lambda = 
 \begin{pmatrix}
  \lambda_{1}  \\
  & \lambda_{2}  \\
   &  & \ddots &   \\
  & &  & \lambda_{n}
 \end{pmatrix} = diag([\lambda_1,...,\lambda_n]) \in \mathbb{R}^{n \times n}
 $$
Usually, the Laplacian matrix we referred is normalized Laplacian.It is easy to see that the Laplacian matrix $L$ associated with an undirected graph is position semi-definite:
Let $f = {f_1,f_2,...,f_n}$ be an arbitray vector, then
$$
\begin{aligned}
f^TLf 
&=f^TDf-f^TWf = \sum^n_{i=1}D_{i,i}f^2_i - \sum^n_{i,j}f_if_jW_{i,j}\\
&= \frac{1}{2}(\sum^n_{i=1}D_{i,i}f^2_i-2\sum^n_{i=1}\sum^n_{j=1}f_if_jW_{i,j}+\sum^n_{j=1}D_{j,j}f^2_j)\\
&=\frac{1}{2}(\sum^n_{i=1}\sum^n_{j=1}W_{i,j}(f_i-f_j)^2) \geq 0
\end{aligned}$$
These are basic facts that simple follow from $L$'s symmetric and positive semi-definite properties:
* $L$ of order $n$ have $n$ linealy independent eigenvectors.
* The eigenvectors corresponding to different eigenvalues of $L$ are orthogonal to each other, and the matrix formed by these orthogonal eigenvectors normalized to the unit norm is an orthogonal matrix 
* The eigenvectors of $L$ can be taken as real vectors
* The eigenvalues of $L$ are nonnegative 
### Laplacian operator
The Laplacian matrix essentially is a Laplacian operator on a graph. To illustrate this concept, we introduce the incidence matrix. The incidence matrix is a matrix that reflect the relationship between vertices and edges.Suppose the direction of each edge in the graph is fixed(but the direction can be set arbitrarily)
Let $f = (f_1, f_2, f_3, ... , f_n)^T$ denote signal vector associated with the vertices $(v_1, v_2,v_3,...,v_n)$, the incidence matrix of a graph, denoted by $\bigtriangledown$, is a $|E| \times |V|$ matrix, the incidence matrix is defined as follows:
$$\bigtriangledown_{i,j}=\begin{cases} \bigtriangledown_{i,j}=-1 & \text{if }v_j \text{ is the intial vertex of edge }e_i\\
\bigtriangledown_{i,j}=1 & \text{if }v_j \text{ is the terminal vertex of edge }e_i\\
\bigtriangledown_{i,j}=0 & \text{if }v_j\text{ is not in }e_i
\end{cases}
$$
The mapping $f \to \bigtriangledown f$ is known as the co-boundary mapping of the graph, we take an example from the graph as follewing, we arrange arbitrary directions to the edges as the figure in right shows. We have
$$\bigtriangledown = 
\begin{bmatrix}
-1 & 0 & 0 & 1 & 0\\
 1 & -1& 0 & 0 & 0\\ 
 0 & 1 & 0 &-1 & 0\\ 
 0 & 0 &-1 & 1 & 0\\ 
 0 & 0 & 0 &-1 & 1\\ 
 0 & 0 & 1 & 0 & -1\\ 
 \end{bmatrix}
$$

Accordingly, $\bigtriangledown 
\begin{bmatrix} f_1\\f_2\\f_3\\f_4\\f_5 \end{bmatrix}=\begin{bmatrix}
-1 & 0 & 0 & 1 & 0\\
 1 & -1& 0 & 0 & 0\\ 
 0 & 1 & 0 &-1 & 0\\ 
 0 & 0 &-1 & 1 & 0\\ 
 0 & 0 & 0 &-1 & 1\\ 
 0 & 0 & 1 & 0 & -1\\ 
 \end{bmatrix}
 \begin{bmatrix} f_1\\f_2\\f_3\\f_4\\f_5 \end{bmatrix}=
 \begin{bmatrix} f_4-f_1\\f_1-f_2\\f_2-f_4\\f_4-f_3\\f_5-f_4\\f_3-f_5\end{bmatrix}$ 
Therefore, $(\bigtriangledown f)(e_{i,j})$ is given by
$$(\bigtriangledown f)(e_{i,j}) = f_i - f_j$$
Furthermore, 
$$\bigtriangledown^T(\bigtriangledown f)=
\begin{bmatrix}
-1 & 1 & 0 & 0 & 0 & 0\\
 1 & -1& 1 & 0 & 0 & 0\\ 
 0 & 0 & 0 &-1 & 0 & 1\\ 
 1 & 0 &-1 & 1 &-1 & 0\\ 
 0 & 0 & 0 & 0 & 1 &-1
 \end{bmatrix}
\begin{bmatrix} f_4-f_1\\f_1-f_2\\f_2-f_4\\f_4-f_3\\f_5-f_4\\f_3-f_5\end{bmatrix}=
\begin{bmatrix} 2f_1-f_2-f_4\\2f_2-f_1-f_4\\2f_3-f_4-f_5\\4f_4-f_1-f_2-f_3-f_5\\2f_5-f_3-f_4\end{bmatrix}$$
Thus, the Laplacian matrix $L$ operating on $g$ would become
$$
\begin{aligned}
Lf=(D-A)f&=
\begin{bmatrix}
 2 & -1& 0 &-1 & 0\\
-1 & 2 & 0 &-1 & 0\\ 
 0 & 0 & 2 &-1 &-1\\ 
-1 &-1 &-1 & 4 &-1\\ 
 0 & 0 &-1 &-1 & 2\\ 
 \end{bmatrix}f=
 \begin{bmatrix} 2f_1-f_2-f_4\\2f_2-f_1-f_4\\2f_3-f_4-f_5\\4f_4-f_1-f_2-f_3-f_5\\2f_5-f_3-f_4\end{bmatrix}\\
&=\bigtriangledown^T(\bigtriangledown f)
\end{aligned}
 $$
Therefore, for any undirected graph,
$$Lf = \bigtriangledown^T(\bigtriangledown f)$$
Particularly, for an $n-$dimensional Euclidean space, the Laplacian operator can be considered as second-order differential operator.
Analogously, consider undirected weighted graphs $G$, each edge $e_{i,j}$ is weighted by $w_{i,j}>0$, the Laplace operator on the graph can be defined as 
$$(Lf)_i = \sum^n_{j=1}=W_{i,j}(f_i-f_j)$$
where $W_{i,j}=0$ if $e_{i,j} \in E$
Also, 
$$(Lf)_i=\sum^n_{j}W_{i,j}(f_i-f_j)=D_{ii}f_i-\sum^n_{j}W_{i,j}f_j=(Df-Wf)_i=(Lf)_i$$
For any $i$ holds, then it can be general form:
$$(D-W)f=Lf$$
As a qudratic form,
$$f^TLf=\frac{1}{2}\sum_{e_{i,j}}W_{i,j}(f_i-f_j)^2$$
Therefore, graph Laplacian matrix $L$, intrinsically as a Laplacian operator, make the centre subtracts the surrounding nodes in turn, multiplying the corresponding link weights at same time, and then sums them.
### Summary
由上都可知，圖學當中定義一個叫做Laplacian Matrix $L$，其簡單定義為$D-A$，或者可以稱為Laplace operator(拉普拉斯算子)，主要為歐幾里得空間中的一個函數的梯度的散度給出的微分算子，通常寫成$\bigtriangledown^2$、$\bigtriangledown . \bigtriangledown$，可以粗糙的視為歐幾里得空間中二次微分操作，總而言之Laplacian就是頂點信息(message)傳播(propagation)。
以單層GCN可以簡單表示為：
$$Y=LX=DX-AX$$
對單一節點來說，式中的$AX$可以視為本次操作(每單位時間)鄰居預計要從我身上拿走的訊息量。 $DX$可以視為是本次操作每個節點尚未進行傳播前，本來擁有的訊息量，即是上次操作(上一個單位時間)每個頂點從它的鄰居頂點取得的訊息量。兩者相減，就是本次操作後訊息傳播的狀況，接著會更近一步說明GCN是如何運用Laplacian Matrix數學理論。

## Spectral graph convolution
在真實世界中，通常每個node都會擁有自己的特徵或屬性，在圖的架構中，我們可以視為圖訊號，可進一步獲取圖相關資訊(如關聯資訊及屬性資訊)，一個圖訊號可以由圖$G = \{V,E\}$所構成的，且可以透過映射函式$f$，來映射節點到數ㄓ值，可以表達為$f:V \rightarrow \mathbb{R}^d$，其中$d$表示與每個節點所關聯數值的維度。接這圖訊號可以透過譜域去解釋，基於graph Fourier transform將圖訊號轉換到不同空間，我並透過Graph Filters進行相關操作，並透過反函式轉換成原始空間，相關操作可以參考下方，
### Graph Fourier transform
Graph Fourier transform is analogous to classical Fourier transform, similarly, the eigenvalues could represent graph frequencies and form spectrum of the graph, the eigenvectors denote frequency components which serve the work as the graph Fourier basis.
Let graph Fourier basis $U = (u_1, u_2, ..., u_n), u_i \in \mathbb{R},i=1,2,...,n$ from Laplacian matrix $L$. Node's singal $f = (f_1, f_2, f_3,...,f_n)^T$, after graph Fourier Transform, signal become $\hat(f)=(\hat{f}(\lambda_1),\hat{f}(\lambda_2),\hat{f}(\lambda_3),...,\hat{f}(\lambda_n))^T$, the graph inverse Fourier tansform is 
$$\hat{f} = U^Tf$$
Correspondingly, Graph Fourier transform is 
$$f = U\hat{f}$$
Therefore, taking Laplace's eigenvector as the basis function , any signal on the graph can be
$$f = \hat{f}(\lambda_1)u_1+\hat{f}(\lambda_2)u_2+...+\hat{f}(\lambda_n)u_n = \sum^n_{i=1}\hat{f}(\lambda_i)u_i$$
where $u_i$ is the column  vector of orthogonal matrix from spectral decompostion from $L=U \Lambda U^T$
In fact, that is analogous to the principle of Discrete Fourier Transform(DFT)
$$X_{2\pi}(k) = \sum_{x = \infty}^\infty x_ne^{-ikn}$$
### Spectral graph convolution 
In the Fourier domain, the convolution operator on graph $\cdot G$ is defined as 
$$g(\cdot G)=F^{-1}(F(g)\odot F(f))=U(U^Tg\odot U^Tf)=Ug_{\theta}(\Lambda)U^Tf=g_{\theta}(L)f$$
where ($\cdot G$) is convolution operator defined on graph, $\odot$ is Hadamard prodoct.It follows that a signal $f$ is filter by $g \in \mathbb{R}^n$, and denoted $g_{\theta}(\Lambda)=diag(U^Tg)$ which the diagonal corresponds to spectral filter coefficients.

For details,
$$
\begin{aligned}
g_{\theta}(\cdot G) f &= g_{\theta}(L) f  = g_{\theta}(U\Lambda U^T) f = Ug_{\theta}(\Lambda)U^T f \\
&=U
    \begin{bmatrix}
    \hat{g}(\lambda_1) & & & \\
    & \hat{g}(\lambda_2) & & \\
    & & \ddots & \\
    & & & \hat{g}(\lambda_n)\\
    \end{bmatrix}U^T f \\
&= U
    \begin{bmatrix}
    \hat{g}(\lambda_1) & & & \\
    & \hat{g}(\lambda_2) & & \\
    & & \ddots & \\
    & & & \hat{g}(\lambda_n)\\
    \end{bmatrix}\hat{f} \\
&= U
    \begin{bmatrix}
    \hat{g}(\lambda_1) & & & \\
    & \hat{g}(\lambda_2) & & \\
    & & \ddots & \\
    & & & \hat{g}(\lambda_n)\\
    \end{bmatrix}
    \begin{bmatrix}
    \hat{f}(\lambda_1) \\
    \hat{f}(\lambda_2) \\
    \cdots \\
    \hat{f}(\lambda_n) \\
     \end{bmatrix} \\
&= U
    \begin{bmatrix}
    \hat{g}(\lambda_1) \\
    \hat{g}(\lambda_2) \\
    \cdots \\
    \hat{g}(\lambda_n) \\
     \end{bmatrix}
    \odot
    \begin{bmatrix}
    \hat{f}(\lambda_1) \\
    \hat{f}(\lambda_2) \\
    \cdots \\
    \hat{f}(\lambda_n) \\
     \end{bmatrix}
\end{aligned}
$$
Spectal-basd GCN all follow this definition of $U g_{\theta}(\Lambda)U^Tf$. the main diference between different version of Spectral-based GCN lies in the choice of the filter $g_{\theta}(\Lambda)$

上述推論，是引入傅立葉轉換方法，在$f$經過Graph filter 進行卷積的到結果，接這我們要來探討不同的Graph filter則會有不同的效果及計算量．
#### Spectral CNN 
Bruna et al. propose the first spectral convolutional neural network. A graph can be associated with node signal $f \in \mathbb{R}_{n \times C_k}$ is a feature matrix with $f_i \in \mathbb{R}_{C_k}$ representing the feature vector of node $i$. A construnction where each layer $k=1,...,K$ transforms an input vector $f^{(k)}$ of size $n \times C_k$ into an output $f^{(k+1)}$ of size $n \times C_{k+1}$
$$f_j^{(k+1)}=\sigma(U\sum^{C_k}_{i=1}g_{\theta_{i,j}}^{(k)}U^Tf_i^{(k)})=\sigma(U\sum^{C_k}_{i=1}g_{\theta_{i,j}}^{(k)}U^T\hat{f}_i^{(k)}) $$
where $g_{\theta_{i,j}}^{(k)},i=1,...,m;j=1,...,C_k$ is a diagonal matrix with trainable parameters $\theta^{(k)}_m,m\in (1,n)$, $\sigma$ is activation function. $g_{\theta_{i,j}}^{(k)}$ is given by 
$$g_{\theta_{i,j}}^{(k)} =
\begin{bmatrix}
\theta_1^{(k)} & & & \\
 & \theta_2^{(k)}& & \\
 & & \ddots& \\
 & & & \theta_n^{(k)}\\
\end{bmatrix} 
$$
上述公式是為第一代GCN設計，主要是在graph filter設計有$n$個$\theta$參數進行訓練，但會導致計算量會龐大，每一次向前傳播會都需要計算$U,diag(\theta)$及$U^T$三者的矩陣相乘，會是有$O(n^3)$計算複雜度．
#### ChebNet
ChebNet uses Chebyshev polynomials instead of convolutions in spectral domain Furthermore, it was demostrated that $g_{\theta}(\Lambda)$ can be approximated by a truncated expansion in terms of Chebyshev polynomials 
$$T_{n+1}(x) = 2xT_n(x)-T_{n-1}(x), n \in \mathbb{N}^+$$
where $T_0(x)=1,T_1(x) = 1$ Here, we make $\tilde{\Lambda} = \frac{2\Lambda}{\lambda_{max}}-I\in[-1,1]$,$\lambda_{max}$ is the biggest eigenvalue from $L$
$$g_{\theta}(\Lambda)= \sum^{K-1}_{k=0}\theta_kT_k(\tilde{\Lambda})$$
where the parameter $\theta \in \mathbb{R}^K$, $T_k(\tilde{\Lambda}) \in \mathbb{R}^{n \times n}$. The filtering operator can also the written as 
$$g_{\theta}(L)= \sum^{K-1}_{k=0}\theta_kT_k(\tilde{L})$$
where $T_k(\tilde{L})\in \mathbb{R}^{n \times n}$ is the Chebyshev polinomial of order $k$ evaluated at the scaled Laplacian $\tilde{L} = 2L/\lambda_{max}-I_n$. Accordingly, spectral filters represented by $K^{th}$-order polynomials of the Laplacian are exactly $K$-localized i.e. it depends only on nodes that are at maximum $K$ steps away from the central node.
##### 比較Spectral CNN 與 CheNet
* Spectral CNN的參數複雜度相對較高，計算複雜度$O(n)$，且較容易overfit，若處理大規模圖資料，則面臨較大的挑戰．
* 計算Laplace的矩陣分解也是非常好計算量
* ChebNet 只有$K$個可訓練參數$\theta_k$，而且$K<<n$，因此複雜度為$O(K)$，複雜度是可以減少的
* ChebNet是不需要進行Laplace的矩陣分解，而是估計$g_{\theta}(L)$，相關也不需要較高計算量．

### GCN 