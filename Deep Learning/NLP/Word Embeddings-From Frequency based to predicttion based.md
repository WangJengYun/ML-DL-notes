Word Embeddings: From Frequency based to predicttion based
===
Word Embeddings是一個常見的手法，主要是針對是將語料庫中的詞(word)轉化為向量的議題，進而可以處理多個應用層面，像是文本分類、情感分析、文章推薦等等，在本文會從傳統的字詞表達到深度學習的中間產物的word2vector，深深了解。

Frequency based Embedding是從corpus中，萃取重要的關鍵字的頻率來表示彼此之間的相關性，在這我們會姊介紹下列幾種方式:
* one-hot representation
* Count Vector
  * TF-IDF vectorization
* Co-Occurrence Matrix

word2vector有了更進一步的發展，work2vector是從大量的文本語料中，以unsupervised learning方式學習語義，並藉由詞向量的表示語義訊息，總而言之，將詞透嵌入空間使語義上的相似詞在空間內的距離相近，是為詞嵌入的一種(word embedding)。word2vector整個建模方式，實際上與自編碼器(auto_encoder)的方法很相近，基於一個神經網絡，此模型訓練好後，不是要得到他的結果，而是這模型透過訓練數據所得到的參數結果，也就是隱藏層的權重矩陣，此現象稱為Fake Task。

本文會介紹下列四種模型:

* google : Continous Bag-of-Words Model 和 Continous Skip-gram Model
* Stanford : GloVe 
* Facebook : Fasttext

## 1. Frequency based Embedding

### 1.1 one-hot representation
過去常見的手法是將它是以one-hot來表達詞向量，每個詞在對應的維度的值為1，其他元素都是0，如下表表示，每一個字以7個維度表示，如果你有10000個字，則會有10000個維度:

word|I|like|deep|learning|NLP|enjoy|flying
----|:----|:----|:----|:----|:----|:----|:----
deep|0|0|1|0|0|0|0
NLP|0|0|0|0|1|0|0
enjoy|0|0|0|0|0|1|0

但這個會有幾個缺點 :
* 每個詞的encoding都會是正交，也就是詞與詞之間會是獨立，無法看出彼此之間的相關性。
* 通常轉換後的維度相當龐大，而且會是稀疏矩陣(sparse matrix)，難以儲存，且這樣做其他的ML的應用，計算量相當複雜而且模型較不穩健(robust)
* 若有新的詞加入語料庫，整個矩陣需要更新

### 1.2 Count Vector

主要考慮一個語料庫(corpus)以 $C$ 代表，包含 $D$ 個文本(Document)，以 $d_1,d2 , ...... ,d_D$ 代表，從 corpus C 擷取N個獨立的詞，然而建構Counter Vector matrix，其中維度為 $D{\times}N$，每一個 row 表示文件 $d_i$ 所包含各個詞的頻率，以下表為例

Document|I|like|deep|learning|NLP|enjoy|flying
----|:----|:----|:----|:----|:----|:----|:----
I like deep learning|1|1|1|1|0|0|0
I like NLP|1|1|0|0|1|0|0
I enjoy flying|1|0|0|0|0|1|1

利用Count Vector可表示每個文本由那些詞所構成以及它的頻率，但是它的缺點如下:

* 需要事前定義詞庫，在真實世界一個語料庫包含上百萬的文本，並可擷取數千萬的詞出來，所以這個Counter Vector matrix 會是 sparse matrix ，難以計算，所以替代方案會是以前10000個頻率較高的詞為主，這個做法需要提前建立所需的詞庫，另外也需要做一些文字處理，像是停字詞(Stop word)
* 必須計算每個詞的頻率。

下如為是是Count Vector matrix 視意圖
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/f9da189d.png?raw=true)

#### 1.2.1 TF-IDF vectorization

在較大的 corpus 中，有些詞是沒有任何意義，像是the,a,is等等，如果我們直接以頻率來判斷富含有意義的詞，這就會忽略較少頻率但是有意義的詞，所以我們會藉由 TF-IDF 轉換來重新定義字詞的權重，轉換次數為數值，這個方式不會只考慮單一個文本出現的頻率，也會考慮整個corpus。

TF-IDF(Term Frequency - Inverse Document Frequency)，主要用於資訊檢索與文字探勘常用的加權技術，為一種統計方式來評估單詞對於文本的集合與詞庫中一份文本的重要程度，定義如下:

Let $t_i$ is the $i^{th}$ term , $d_j$ is the $j^{th}$ document and $D$ is the number of documents

**Term Frequency (tf)** : gives us the frequency of the word in each document in the corpus
$$TF_{t_id_j} = \frac{Number\,of\,times\,the\,term\,appear\,in\,document\,d_j}{Total\,number\,of\,term\,in\,document\,d_j} = \frac{n_{t_id_j}}{\sum_{k=1}n_{t_kd_j}}$$

**Inverse Data Frequency (idf)**: used to calculate the weight of rare words across all
$$IDF_{t_id_j} = \log(\frac{Total\,number\,of\,documents}{Number\,documents\,with\,term\,in\,it}) = \log(\frac{|D|}{|{d_j:t_i\in d_j}|})$$
**TF-IDF score (w)** for a word in a document in the corpus
$$IFIDF_{t_id_j} = TF_{t_id_j} \times IDF_{t_id_j} $$

視意圖如下:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/b99f1c7a.png?raw=true)

##### 1.2.2 Co-Occurrence Matrix

接下探討詞與詞的相似程度，通常較相似的詞都會傾向一起出現，像是 Apple is a fruit and Mango is a fruit. 這兩句的呈現可以說apple與Mango是相似詞。

由上敘述的概念，進一步定義如何計算相似程度，通常建立Co-Occurrence Matrix，而且會在一個window計算共同出現次，首先讓$count(w(current)|w(surround))$為有多少次的W(currrent)在給定周圍的W(surround)，我們藉由這個定義可以建立Co-Occurrence Matrix。假設
$$Corpus = I\,like\,deep\,learning,I\,like\,NLP,I\,enjoy\,flying$$

假設windows為2，來建立Co-Occurencematrix:
 __|I|like|deep|learning|NLP|enjoy|flying
----|:----|:----|:----|:----|:----|:----|:----
I|0|**3**|2|1|2|1|1
like|1|0|1|1|1|0|0
deep|2|1|0|1|0|0|0
learning|1|1|1|0|0|0|0
NLP|2|1|0|0|0|1|0
enjoy|1|0|0|0|1|0|1
flying|1|0|0|0|0|1|0

給定前後兩個詞有包含like詞，計算I出現的次數:
* **I** _like_ deep learning,I like NLP,I enjoy flying
* I like deep learning,**I** _like_ NLP,I enjoy flying
* I like deep learning,I _like_ NLP,**I** enjoy flying

如果在一個Corpus有V個詞，則Co-Occurrence Matrix的維度會是 $V \times V$，會是有兩種情況:
* 如果你得詞彙V很大量，則會是很困難處理
* 必須要處理停字詞(Stopword)，將這些無意義的處理掉，但即使這樣V也還是很大量

另外要注意的是，Co-Occurrence Matrix不是詞向量表達，然而我們可以藉由PCA、SVD等等可以降維的方式，找數個factor來表示詞向量，下圖是介紹SVD:

可以藉由SVD矩陣分解，並選取少少的eigenvector與eigenvalue來表示Co-Occurrence Matrix，而切不失去大部分的資訊。
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/f0d92bf1.png?raw=true)

結論為:
* 此方法可以表達每個詞之間的語義(semantic)的相關性
* 藉由SVD產生詞向量，是屬於一次性計算，相對比較快
* 他需要較大記憶體儲存Co-Occurrence Matrix

## 2. Prediction based Embedding

此方法是基於深度學習的神經網絡的產物，可以透過Fake Task來訓練資料並得到詞向量，向量本身可計算相似程度(word similarity)與字詞的比喻(word analogy)，像是King-man+woman = Queen，這會是相當神奇的。

### 2.1 google : Continous Bag-of-Words Model and Continous Skip-gram Model

#### 2.1.1 n-gram model

在介紹這兩個方法之前，先了解語言模型(language model)，此模型在一串token的序列定義一個機率分配，主要還是看模型如何設計，其中token可以是字詞、字元甚至byte，然而最早的語言模型是基於固定長度的token，稱為n-gram model 

n-gram model是給定先前的 $n-1$個token的$n^{th}$個條件機率，機率分配如下:

$$P(x_1,x_2,...,x_{\tau})=P(x_1,...,x_{n-1})\prod_{t=n}^\tau{P(x_t|x_{t-n+1},...,x_{t-1})}$$

對於特性的n，有不同的模型，如下:
* unigram(n=1):
$$P(x_1,x_2,...,x_{m})=\prod_{i=1}^m{P(x_i)}$$
* bigram(n=2):
$$P(x_1,x_2,...,x_{m})=\prod_{i=1}^m{P(x_i|x_{i-1})}$$
* trigram(n=3)
$$P(x_1,x_2,...,x_{m})=\prod_{i=1}^m{P(x_i|x_{i-1},x_{i-2})}$$

套用在NLP的例子上，就會是像給定前幾詞，下一個特定的詞的機率，假設
$$Corpus = I\,like\,deep\,learning$$
計算"I like deep learning"的機率，以bigram(n=2)model為例，如下 :
$$P(I like deep learning) = P(I)P(like|I)P(deep|like)P(learning|deep)$$
計算機率可以由Co-Occurrence Matrix而來，這裡就不說明了。

#### 2.1.2 Feedforwoard Neural Language Model (FNLM)

基於n-gram的概念，接著了解NNLM的語言模型，因為CBOW與skip-gram是由NNLM調整而來，這個模型是基於n-gram的衍生，勝制比她更好，此方法主要的特性如下:
* 以word vector feature考慮詞庫裡的每個詞之間的相關性
* 以word vector feature表達詞序列的聯合機率函數(joint probablity function)
* 同時學習word vector feature與機率函數的參數


先從整理來看模型，讓$w_1,...,w_T$為詞序列，其中$W_t\in{V}$，$V$為詞彙空間，較大且有限集合，此模型的機率函數表示為$f(w_t,...,w_{t-n+1})= \hat{P}(w_t|w^{t-1}_1)$，模型會分為兩的部分:
1. 我們會有一個$|V| \times m$映射矩陣$C$，這個矩陣表達詞向量，另外$C(i)$為第i的詞向量
2. 接著我會有g函式映射輸入的詞向量為$(C(w_{t-n+1}),...,C(w_t-1))$至給定輸入的詞$(w_{t-1},...,w_{t-n+1})$下,$w_{t}$的條件機率，然而g的輸出為向量，其中第$i^{th}$元素估計$\hat{P}(w_t=i|w^{t-1}_1)$，整體公式表達為:
$$f(i,w_{t-1},...,w_{t-n+1})=g(i,C(w_{t-1},...,C(w_{t-n+1})))$$

然而$f$會是有兩個映射，$C$所有詞的共同分享的權重，g其實視神經網絡，而他的參數表示為$\omega$，則全部的參數表示為 $\theta=(C,\omega)$。

loss function:
$$L=\frac{1}{T}\sum_t{log(f(w_t,w_{t-1},...,w_{t-n+1};\theta))}+R(\theta)$$
其中R($\theta$)為正規項(regularization)

接下來我們細看每一層，我們主要輸入詞向量為 $x = (C(w_{t-1}),C(w_{t-2}),...,C(w_{t-n+1}))$，輸入一個神經網絡:

$$y = b+Wx+U+tanh(d+Hx)$$

最後輸出層需要透過softmax轉換，能夠保證機率和為1，如下:

$$\hat{P}(w_t|w_{t-1},...,w_{t-n+1}) = \frac{e^{y_{w_t}}}{\sum_i{e^{y_i}}}$$

此模型架構如下

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/e429afb0.png?raw=true)

但是Feedforwoard Neural Language Model，還是有他的缺點:
* 訓練會受到詞序列的大小，若是以padding方式來輸入較多的序列，則會難以訓練
* 此模型還是較難抓到詞之間的相關程度，只能訓練給定前幾個詞預測下詞的機率，無法更精準知道詞之間的相關程度。

#### 2.1.3 CBOW 和 Skip-gram 初步想法

Work2vector是google從Feedforwoard Neural Language Model調整架構所衍生出來，並提取其中wieght表示詞向量(如下圖表示)，其詞向量表示自詞之間的相關程度(如下圖)，進而可以以線性組合方式猜測可能的值，如$King-man+woman = Queen$，，總而言之，就是將一段多詞組成的文本，轉換成詞向量表示，並且數值化的資料，送到模型做後續的應用。

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/2dc5d7bb.png?raw=true)

接下來我們會介紹Continous Bag-of-Words Model 和 Continous Skip-gram Model，他之間差異如下:
* CBOW : 模型輸入前後詞(contex word)詞向量，要預測出可能的目標詞(target word)；
* Skip-gram : 模型則是輸入目標詞詞向量，預測出可能的前後詞。

此兩個模型架構，如下圖 : 
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/23a1fae6.png?raw=true)

以下列的windows設為2為例:
![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/223a0104.png?raw=true)

 CBOW|Skip-gram|
----|:----|
[ quick , brown ] => The|The => [ quick , brown ]
[ the , brown , fox ] => quick|quick => [ the , brown , fox ]
[ the , quick , fox , jumps ] => brown|brown => [ the , quick , fox , jumps ]
[ quick , brown , jumps , over ] => fox|fox => [ quick , brown , jumps , over ]
PS : => 左邊為input，右邊為output。

有此例子可知，訓練樣本是由文本而來的，再放入神經網絡中，來產生詞向量，此訓練過程可以歸為非監督學習(unsupervised learning)，接下來一一介紹 CBOW 和 Skip-gram

#### 2.1.4 Continous Bag-of-Words Model
##### 2.1.3.1 One-word context
首先介紹最簡單的例子，先假設藉由目標詞前後兩個詞估計目標詞出現的機率，這其實就是bigram model，進而建構Simple CBOW。

Assume:
* $V$ is the vocabulary size (詞彙個數)
* $N$is the hidden layer size (隱藏層的個數)
* $W$ is the $V \times N$ matrix which is weigth between the input layer and the hidden layer(為輸入層至隱藏層的權重)
* h is hidden laryer
* $W^{'}$ is the $N \times V$ which is weigth between the hidden layer and the output layer(為隱藏層至輸出層的權重)
* The input is a one-hot encoded vector and  only one out of V units, ${x_1,...x_V} will be 1,and all other units are 0.

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/a0532151.png?raw=true)

再輸入層至隱藏層中，每個詞向量表示如下:

Each row of $W$ is the $N$-dimension vector $V_w$ of associated word of input layer.For row i of $W$ is $v^{T}_{w}$, and  given a word,assuming $x_i = 1$ and $x_j = 0$ for $i \neq j$ : 
$$ h = W^Tx=W^T_{(i,.)} := v^T_{w_i}$$
以下列為視意圖，表示可透過 one-hot vector乘上$W$可得到每個詞的權重(向量):

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/495a301e.png?raw=true)

接下來，隱藏層至輸出層，可以由下列表示:

Let $v^{'}_{w_j}$ is the $j^{th}$ column of the matrix $W^{'}$ 
$$u_j = (v^{'}_{w_j})^Th$$
Then we can use sofemax,a log-linear classification model to obtain posterior distribution of words. The multinomial distribution : 
$$p(w_j|w_i) = y_j = \frac{exp(u_j)}{\sum_{j^{'}}^V{exp(u_{j^{'}})}} = \frac{exp((v_{w_j^{'}}^{'})^Tv_{w_i})}{\sum_{j^{'}}^V{exp((v_{w_{j^{'}}^{'}}^{'})^Tv_{w_i})}}$$
where $y_j$ is the output of the $j^{th}$ unit in the output layer,and $v_w$ and $v_{w}^{'}$ are representation of the word w,so we call $v_w$ as the input vector and $v_w^{'}$ as the output vector of the word $w$.

接著希望可以maximize機率，如下: 

Maximize the conditional probability of boserving the word $W_O$ given the input context word $W_I$ with regurad to the weight: 
$$max\,p(W_O|W_I) = max\,y_{j^*} = max\,log y_{j^*} = u_{j^*}-log\sum_{j^{'}=1}^V{exp(u_{j^{'}})} := -E$$
where $E = -log\,p(W_o|W_I)$ is loss function and $j^*$ is the index of the autual output word in the output layrer.


##### 2.1.3.2 Multi-word context

經過backpropagation更新參數，達到多次迭代讓模型達到穩定，此部分讓大家了解再輸入為一個單詞時，是如何定義，接著輸入為多的單詞，進行訓練此模型架構如下:

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/370e77e9.png?raw=true)

由上圖結構可知道，再輸入層會有多個前後詞，則會共用相同weight，進行訓練，然而輸入層至隱藏層中，每個詞向量表示如下:

$$h = \frac{1}{C}W^T(x_1+x_2+...+x_C)= \frac{1}{C}(v_{w_1},v_{w_2},...,v_{w_C})^T$$
where C is number of words in the context $w_1,...,w_C$ are the words in the context.
The loss fucntion : 
$$loss\,function= -log\,p(w_O|w_{I,1},...,w_{I,C}) = -u_{j*} + log\sum_{j^{'}=1}^V{exp(u_{j^{'}})}=-(V^{'}_{w_O})^Th+loglog\sum_{j^{'}=1}^V{exp((v_{w_j}^{'})^Th)}$$

與One-word context不同是h的計算方式。

#### 2.1.4 Skip-gram model 

此模型的架構與CBOW的想法相反，是想利用目標詞預測前後詞，模型架構如下:

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/9867cd41.png?raw=true)

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/82463224.png?raw=true)

因為只input只有目標詞，所以$h$，會是與one-word context的CBOW一樣，如下:
$$h = W^T_{(i,.)}:=V^T_{w_I}$$

因為輸出層變多個詞，隱藏層至輸出層修改如下:
we are outputing C multinomial distribution.
$$p(W_{c,j}=W_O|W_I)=y_{c,j}=\frac{exp(u_{c,j})}{\sum_{j^{'}=1}^V{exp(u_{j^{'}})}}$$
where 
* $w_{c,j}$ is the $j^{th}$ word on the $c^{th}$ panel of output layer.
* $u_{c,j}$ is the net input of the $j^{th}$ unit on the $c^{th}$ panel of the output layer
然而對於前後詞適用共相同的weight:

the output layer panels share the same weights,so:
$$u_{c,j} = u_j = v^T_{w_j}h,\,\,\,\,\, for\,\,c = 1,2,3,...,C$$
where $V^{'}_{w_j}$ is the output of $j^{th}$ word in the vocabulary.

The loss function :
$$E = -log\,\,p(W_{O,1},W_{O,2},W_{O,3},...,W_{O,C}|W_I) = -log\prod_{c=1}^C\frac{exp(u_{c,j^*_c})}{\sum_{j^{'}=1}^V{exp(u_{j^{'}})}}=-\sum_{c=1}^C{u_{j^*_c}}+C\,log\sum_{j^{'}=1}^V{exp(u_{j^{'}})}$$

where $j_c^{'}$ is the index of actual $c^{th}$ output context word in the vocabulary. 

#### 2.1.5 Optimizing Computational Efficiency

接下來我們要來介紹如何可以加快訓練詞向量，假設我們有$V$個詞彙，在每次迭代都要透過更新$V$的詞彙的Weight，然而計算softmax得到各詞彙的機率，相對需要很花費時間，針對這個問題，提出兩個方法可以改善，如下:
* Hierarchical Softmax
* Negative Sampling 

##### 2.1.5.1 Hierarchical Softmax

此方法是非常有效率的計算softmax，在2005與2009年所提出，主要是以二元數(binary tree)來代表詞彙的所有詞，假設我們有$V$個詞彙，則會有$V$個葉節點(leaf node)與內部會有$V-1$節點，對於每個葉節點從根節點到葉節點都會存在著一個唯一的路徑，此路徑就會估計這個詞的出現機率，架構如下:

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/6994a2cf.png?raw=true)

在這個Hierarchical Softmax，是沒有輸出的詞向量，因為在每一個內部 $V-1$節點有輸出向量 $V^{'}_{n(w,j)}$，對於目標詞的機率，定義為:
$$p(w=w_o)=\prod^{L(w)-1}_{j=1}{\sigma ([n(w,j-1)==ch(n(w,j))](V^{'}_{n(w,j)})^Th)}$$

where 
* $ch(n)$ is the left child of unit n
* $V^{'}_{n(w,j)}$ is the vector representation(output vector) of the inner unit $n(w,j)$
* $h$ $is the output value of the hidden layer 
  * skip-gram model : $h = V_{W_I}$
  * CBOW : $h = \frac{1}{C}\sum^C_{c=1}{V_{w_c}}$
* $[x] = \left\{\begin{array}{l}1\,\,\,if \,x \,is\,true \\-1 \,\,\,otherwise\end{array}\right .$
* $\sigma(x) = \frac{1}{1+exp(x)}$

二元數有左邊與右邊，通常會以random work的機率來表達，在內部的節點會有走左邊的機率與右邊的機率，定義如下:
* The probability of going left at unit n : $p(n,left) = \sigma((V^{'}_n)^Th)$
* The probability of going right at unit n : $p(n,right) = 1 - \sigma((V^{'}_n)^Th) = \sigma(-(V^{'}_n)^Th)$

根據上圖為例，我們可以計算從根節點至$W_2$的葉節點的機率，定義如下:
$$p(w_2 = W_O) = p(n(w_2,1),left) \cdot p(n(w_2,2),left) \cdot p(n(w_2,3),right) =  \sigma((V^{'}_{n(w_2,1)})^Th) \cdot \sigma((V^{'}_{n(w_2,2)})^Th)\cdot \sigma(-(V^{'}_{n(w_2,3)})^Th)$$
然而會有很好的性質是Hierarchical Softmax在全部詞定義為多項分配(multinormial distribution):
$$\sum^V_{i=1}{p(w_i = w_O) = 1}$$
最後我們得到loss function，定義為:
$$E = -log\,p(w=w_O|w_I) = -\sum^{L(W)-1}_{j=1}{log\,\sigma ([n(w,j-1)==ch(n(w,j))](V^{'}_{n(w,j)})^Th)}$$

##### 2.1.5.2 Negative Sampling 

此方法比Hierarchical Softmax更直接的解決問題，因為詞彙較大，較難處理，每次迭代更新outputvector相當麻煩，則採用抽樣的方式，且只更新抽樣的部分。然而在抽樣過程中，顯然是要保留output word(ground truth)，且也需要一些樣本視為負樣本來訓練模型，paper實驗數據說明，大數據使用2至5個負樣本;小數據使用5至20個負樣本，然而loss functon 定義如下:

$$E = -log\,\sigma((V^{'}_{W_O})^Th) - \sum_{w_j \in W_{neg}}{log\sigma(-(V^{'}_{w_j})^Th)}$$
where 
* $W_O$ is the output word (i.e.,the positive samples)
* $V^{'}_{w_O} is the output vector$
* $h$ $is the output value of the hidden layer 
  * skip-gram model : $h = V_{W_I}$
  * CBOW : $h = \frac{1}{C}\sum^C_{c=1}{V_{w_c}}$
* $W_{neg}= \{w_j|j=1,...,K\}$ is the set of words that are sampled basd on $P_n(w)$ (i.e., negative samples)

接著了解 noise distribution $P_n(w)$視為抽樣的機率分配，定義如下:
$$P_n(w) = (\frac{U(w)}{Z})^{\alpha}$$
where 
* $U(w) is the unigram distribution$
* $Z$ is the normalization factor 

$\alpha$主要會影響分配的平滑化，可藉由調整來減少抽到一般詞的機率或增加較少出現的詞的機率，來解決一般詞與較少出現的詞的不平衡現象，在paper中 $\alpha$為以$\frac{3}{4}$視為最好的結果。

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/f0dc3d24.png?raw=true)

##### 2.1.5.3 Subsampling of frequent words

在大量的文本中，有些詞是出現頻率很高但是沒有富含任何意義，也無法幫助模型訓練出詞的意涵，像是 the、a 等等，這些訊息量不如低頻詞，所以為了解決此問題，提出保留字詞的機率如下:
$$P(w_i) = (\sqrt{\frac{z(w_i)}{t}}+1)\cdot\frac{t}{z(w_i)}$$
where 
* $t$ is a chosen threshold , default $10^-5$
* $z(w_i)$ is the frequency of word $w_i$

![](https://github.com/WangJengYun/ML-DL-notes/blob/master/image/Word%20Embeddings%20From%20Frequency%20based%20to%20predicttion%20based/8dbfd644.png?raw=true)

在這個部分會有幾個有趣的點:
* $P(w_i) = 1.0$((100% chance of being kept) when $z(w_i)<=0.0026$
  * 這個意思為這個詞出現的比例為0.0026將會被二次抽樣
* $P(w_i) = 0.5$((0.5% chance of being kept) when $z(w_i)<=0.00746$
* $P(w_i) = 0.033$((3.3% chance of being kept) when $z(w_i)<=1.0$
  * 這個詞在所有文本都會出現

