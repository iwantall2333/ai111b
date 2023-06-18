# 目錄 :

- Neural Network

  - Activation Function

  - Loss Function

  - 反向傳播

  - Softmax

- CNN

- RNN

- ELMo、BERT 以及 GPT

- 孿生神經網路（Siamese Network）

- Transfer Learning

- [AutoEncoder](#autoencoder)

- [GNN](#gnn)

- Cycle GAN

- [Transformer](#transformer)

- gpt?

- 對資料的正規化 (?)

- [NLP](#nlp)

  - 數據前處理

  - 資料形式說明

  - Word Embedding

- [yolo](#yolo)

- [機器學習任務](#機器學習任務)

- [實作概念](#實作概念)

---

# 前置閱讀材料

- [給所有人的深度學習入門：直觀理解神經網路與線性代數@Meng Lee](https://leemeng.tw/deep-learning-for-everyone-understand-neural-net-and-linear-algebra.html) 2019-10-13; 20221130

  - 很值得看，從基本數學入門到神經網路，直覺地理解神經網路架構意義

  - 比如加入激勵函數（activation function）讓 FC 層跟其他層之間有**非線性的轉換**，目的是讓神經網路掌握超越線性轉換的能力的動畫演示 :
    <video id="video" controls="" preload="none" poster="ReLu"  controls loop autoplay> <source id="mp4" src="https://leemeng.tw/images/manim/twolayersreluInbetweensolvehardtwocurves.mp4" type="video/mp4"></videos>

    影片來源 : https://leemeng.tw/images/manim/twolayersreluInbetweensolvehardtwocurves.mp4

# Neural Network

類神經網路 (模擬人類大腦運作的方式)

#### Activation Function

激勵函數

把計算好了的輸入值standardize 好，**規範它的「輸出數值之範圍」，「輸出數值的相互關係」**。 ( 就是一個輸入x 得出y 的算式 )

> 幾種常用的Activation Function

![各激勵函數圖型From medium.com](https://miro.medium.com/max/828/1*ACHo09NFhKvYCsOFHxWVbA.png)

1. ReLU (Rectified Linear Units) */ˋrɛlu/*

    - 把負數先轉為零，正數就什麼都不做

2. Leaky ReLU

    - 就是不想放棄那些負數，只是負數的影響力小一點。例如把負數乘0.1 或乘 0.01

3. ELU

    - 進一步把Leaky ReLU 跟 ReLU 的硬朗線條改smooth 一點

4. Sigmoid

    - 能將非常大的負值變成接近0，非常大的正值變成接近1，而在取值0附近則是平穩成長。

    - 輸出意義 :

      1. 負正分界於0.5，更使其在NN 界中用作**二元判斷**的好工具(判斷非黑即白）。

      2. output數例如0.75，**也可以當成是prediction的概率**，75%。

    - 參考資料 :

      - [3Blue1Brown series  第 3 季 第 1 集究竟神經網路是什麼？ l 第一章 深度學習](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1)

5. tanh

    - 與Sigmoid做比較，tanh 的輸出範圍是由-1 至，Sigmoid是0 ~ 1 (意義是甚麼?)

6. Maxout

    - 在相鄰兩個neurons 先算好 x+y-b ，取較大值作為輸出。


參考資料 :  

1. [【ML09】Activation Function 是什麼?@Tim Wong](https://medium.com/%E6%B7%B1%E6%80%9D%E5%BF%83%E6%80%9D/ml08-activation-function-%E6%98%AF%E4%BB%80%E9%BA%BC-15ec78fa1ce4)

    - 有說明輸出數值的意義

2. [給所有人的深度學習入門：直觀理解神經網路與線性代數@Meng Lee](https://leemeng.tw/deep-learning-for-everyone-understand-neural-net-and-linear-algebra.html) 2019-10-13; 20221130


    - 有用影片說明加入激勵函數（activation function）的目的 : 讓 FC 層跟其他層之間有**非線性的轉換**，讓神經網路掌握超越線性轉換 ( 矩陣 ) 的能力

#### 損失函數 Loss Function

給定一個 正確解答 以及 模型預測的結果，我們的**模型會透過損失函數就能自動計算出現在的預測結果跟正解的差距為多少**。

內文來源 :

1. [進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南@leemeng](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E6%B1%BA%E5%AE%9A%E5%A6%82%E4%BD%95%E8%A1%A1%E9%87%8F%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%A1%A8%E7%8F%BE) 2018-12-24;20221227


#### 優化器 Optimizer

有了 Loss Function 給出的 預測結果跟正解的差距，
接著就要透過 Optimizer 來持續修正模型裡頭的參數，以達到最小化損失函數

不同optimizer :
![](https://leemeng.tw/images/nlp-kaggle-intro/loss-function-learning.gif)
git來源 : https://leemeng.tw/images/nlp-kaggle-intro/loss-function-learning.gif

雖然我們有很多種optimizer，但它們基本上都是從梯度下降法（Gradient Descent）延伸而來

1. Adam

    - [【ADAM算法 论文精读】史上最火梯度下降算法是如何炼成的？](https://www.bilibili.com/video/BV1Sg41197kL/?spm_id_from=333.788.recommend_more_video.0&vd_source=c2cc9cbcc46ca21aa3ac624cde210a9f)

1. SGD optimizer

    - [模型每看完 1 個訓練數據就嘗試更新權重，而因為單一一筆訓練數據並不能很好地代表整個訓練資料集，前進的方向非常不穩定。](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E8%A8%93%E7%B7%B4%E6%A8%A1%E5%9E%8B%E4%B8%A6%E6%8C%91%E9%81%B8%E6%9C%80%E5%A5%BD%E7%9A%84%E7%B5%90%E6%9E%9C)

- 待看:https://blog.csdn.net/S20144144/article/details/103417502

References :

1. [進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南@leemeng](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E6%B1%BA%E5%AE%9A%E5%A6%82%E4%BD%95%E8%A1%A1%E9%87%8F%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%A1%A8%E7%8F%BE) 2018-12-24;20221227

#### 反向傳播

參考資料 :

1. [Backpropagation calculus | Chapter 4, Deep learning@3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4) 2017年11月3日 ; 20221224 0.5倍速


#### Softmax

Softmax 函式一般都會被用在整個神經網路的最後一層上面，比方說我們這次的全連接層。

Softmax 函式能將某層中的所有神經元裡頭的數字作正規化（Normalization）：將它們全部壓縮到 0 到 1 之間的範圍，並讓它們的和等於 1。

1. 所有數值都位於 0 到 1 之間
2. 所有數值相加等於 1

這兩個條件恰好是機率（Probability）的定義，Softmax 函式的運算結果可以讓我們將每個神經元的值解釋為對應分類（Class）的發生機率。

內文來源 :

1. [進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南@leemeng](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#3-%E9%96%80%E6%8E%A8%E8%96%A6%E7%9A%84%E7%B7%9A%E4%B8%8A%E8%AA%B2%E7%A8%8B) 2018-12-24;20221227


---

# [CNN](https://zhuanlan.zhihu.com/p/27908027)

  - 卷积层


>ccc:梯度
>- 即微分，而微分後會得到斜率
>- 函數圖型上每個點都會得到一個的斜率(機器學習中利用斜率作為**向量**，該向量可能為增長方向可能為向下方向)
>
>梯度下降
>-  **產生向量後 每次往該向量走**，再微分，再往該方向(向上或向下走)，不斷重複 直到向量是0走不動，就會走到最高點或最低點(終點)
>斜率越垂直，走的幅度越大 (可以想像一下)
>
>![維基百科梯度下降示意圖片](https://upload.wikimedia.org/wikipedia/commons/d/d2/3d-gradient-cos.svg)
>下面藍色的箭頭是上面對應點的斜率(有方向性)
>
>梯度下降法很像爬山演算法，但是方向往逆梯度方向走

參考資料 :

1. [Backpropagation calculus | Chapter 4, Deep learning@3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4) 2017年11月3日 ; 20221224 0.5倍速


>ccc:捲積 : 原圖 + 捲積和 ---> 新圖層
>
>捲積相對多層感知機的好處 : 為捲積和共享 加上 不用全連接層 參數少很多
>


  - Relu层
  - 池化层
  - 全连接层
    - Softmax : 它是一个分类函数

參考資料 :

- [Classify MNIST digits with a Convolutional Neural Network](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)

  - 最後的softmax就是把明顯的變更明顯，不明顯變更不明顯 (ccc看著維基百科的公式說，即把東西成指數性放大縮小↓)
    >![](https://wikimedia.org/api/rest_v1/media/math/render/svg/e348290cf48ddbb6e9a6ef4e39363568b67c09d3)

# RNN

簡單理解的話，RNN就是Neural Network 加上 memory機制

每個字都帶有一個向量，[我,祖,你,孫] 依順序輸入進去 :

輸入[我]之後，RNN內運行的樣子 :

```markdown
y1 = 0, y2 = 1,
m1,m2,m3原本為空值

--> b1 = y1*1 + y2*1, b2 = y1*1 + y2*1, b3 = y1*1 + y2*1, --> b1 = 1,b2 = 1,b3 = 1
--> g1 = b1+b2+b3 = 3, g2 = b1+b2+b3 = 3
--> m1=b1 = 1, m2=b2 = 1, m3=b3 = 1  
```

接著輸入[祖]之後，RNN內運行的樣子 :

```markdown
y1 = 1, y2 = 0,
m1 = 1,m2 = 1,m3 = 1 //即memory會記住上一次 b1 b2 b3的結果

--> b1 = y1*1 + y2*1 + (m1*1 + m2*1 + m3*1), b2 = y1*1 + y2*1 + (m1*1 + m2*1 + m3*1), b3 = y1*1 + y2*1 + (m1*1 + m2*1 + m3*1) --> b1 = 1, b2 = 1, b3 = 1
//memory會記住上一次 b1 b2 b3的結果，來影響這一次的b1 b2 b3

--> g1 = b1+b2+b3 = 12, g2 = b1+b2+b3 = 12
--> m1=b1 = 4, m2=b2 = 4, m3=b3 = 4 // memory記住這次的b1 b2 b3，來影響下一次運行
```

後續輸入[你,孫]也是一樣的步驟。

這代表 :

1. 「你祖我孫」與「我祖你孫」兩個句子經過RNN分析後，輸出的結果可能是不同的。

2. 這次輸出的結果會受到上一次、上上次、上上上次輸出的結果影響 ; 即**輸入的順序**不同，會影響輸出的結果



參考資料 :

1. [10分鐘了解RNN的基本概念@李政軒](https://www.youtube.com/watch?v=6AW80qmaAOk)2020年12月6日;20221225

    - 說明RNN只是Neural Network加上memory

    - 說明RNN中memory是怎麼一步步運作的

2. [進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南@leemeng](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#3-%E9%96%80%E6%8E%A8%E8%96%A6%E7%9A%84%E7%B7%9A%E4%B8%8A%E8%AA%B2%E7%A8%8B)2018-12-24;20221225

## RNN缺陷

重新看剛剛上方的程式碼，關於memory中的值影響輸出的那段 :

```markdown
--> b1 = y1*1 + y2*1 + (m1*1 + m2*1 + m3*1), b2 = y1*1 + y2*1 + (m1*1 + m2*1 + m3*1), b3 = y1*1 + y2*1 + (m1*1 + m2*1 + m3*1) --> b1 = 1, b2 = 1, b3 = 1
//memory記住上一次 b1 b2 b3的結果，來影響這一次的b1 b2 b3
```

上方的例子memory往b1、b2、b3的權重是1，
假如權重是0.5，第一次放入memory的值 r1 ，會在下一次加到b1、b2、b3之前被乘以0.5，
再下一輪 r1 又會再被乘以0.5然後加到b1、b2、b3，
不斷進行下一輪， r1 就會接近0，相當於 r1 對越後面的輸入影響越小。(又稱梯度消失、梯度彌散)

- **假如權重 < 1**，RNN 在處理較長序列的輸入資料時，會忘記「更早以前」發生的事情。

**如果權重 > 1**，則會使 r1 對後面輸入的影響越來越大，到最後RNN被r1撐死 。(又稱梯度爆炸)

參考資料 :

1. [什么是 LSTM RNN 循环神经网络 (深度学习)? What is LSTM in RNN (deep learning)?@莫烦Python](https://www.youtube.com/watch?v=Vdg5zlZAXnU&t=2s)2016年9月30日;20221226



#### time step

抓序列數據中，前n筆資料，一起來預測下一筆資料。 ( 應該是這個意思?? )

比如序列ABCD...，time_step = 2，代表會用"CD"來預測下一筆資料。
time_step = 3，代表會用"BCD"來預測下一筆資料。

而序列數據可以是**時間序列**，代表可以做比如『動作軌跡』類型等...與連續時間有關的東西

參考資料 :

1. [LSTM的time steps到底有什么意义？](https://www.zhihu.com/question/264805289)


## 實作


建立一個 RNN layer：

```python
from keras import layers
rnn = layers.SimpleRNN()
# 利用keras就可以建立一個 RNN layer
```


參考資料 :

1. [進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南@leemeng](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#3-%E9%96%80%E6%8E%A8%E8%96%A6%E7%9A%84%E7%B7%9A%E4%B8%8A%E8%AA%B2%E7%A8%8B)2018-12-24;20221225

### Forget gate

f<sub>t</sub>  

- f 的算法 : x<sub>t</sub> 與 h<sub>t-1</sub> 合併，再與 W<sub>f</sub> 做乘積，最後做 sigmoid 使 f 的每個元素介於 0~1 之間

接著

- C 。 f

  **<font color =LightSalmon>f 有選擇地讓 C<sub>t-1</sub> 的值通過</font>** :

  ```markdown
  [0.9 0.2 -0.5] 。 [0.5 0 1] --> [0.45 0 -0.5]
  // f 元素是 0，則C對應的 0.2 就不能通過
  // f 元素是 1，則C對應的 -0.5 就全部通過
  ```


### Input gate & New Value

input gate : i<sub>t</sub>

- i<sub>t</sub> 的算法 : x<sub>t</sub> 與 h<sub>t-1</sub> 合併，再與參數矩陣 W<sub>i</sub> 做乘積，最後做 sigmoid 使 i<sub>t</sub> 的每個元素介於 0~1 之間

New Value : <sup>~</sup>C<sub>t</sub>

- <sup>~</sup>C<sub>t</sub> 的算法 : x<sub>t</sub> 與 h<sub>t-1</sub> 合併，再與參數矩陣 W<sub>c</sub> 做乘積，最後做 tanh 使 <sup>~</sup>C<sub>t</sub> 的每個元素介於 -1 ~ 1 之間

C 。 f 後接著 **<font color =LightSalmon>+【i<sub>t</sub> 。 <sup>~</sup>C<sub>t</sub>】以添加新的信息</font>**，這樣就更新完C<sub>t</sub>了

### Output gate

o<sub>t</sub>

- o<sub>t</sub> 的算法 : x<sub>t</sub> 與 h<sub>t-1</sub> 合併，再與參數矩陣 W<sub>o</sub> 做乘積，最後做 sigmoid 使 o<sub>t</sub> 的每個元素介於 0~1 之間

最後 h<sub>t</sub> 的算法 : o<sub>t</sub> 。 tanh( 更新完的C<sub>t</sub> )

h<sub>t</sub> 傳給 LSTM 的輸出、並存到 momery 中給第 t+1 個輸入當參數


參考資料 :

1. [RNN模型与NLP应用(4/9)：LSTM模型@Shusen Wang](https://www.youtube.com/watch?v=vTouAvxlphc)2020年3月22日;20221226

    - 搭配[github上的ppt](https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_3.pdf)

## 實作


建立一個 LSTM layer：

```python
from keras import layers
lstm = layers.LSTM()
# 利用keras就可以建立一個 LSTM layer
```

# GRU

LSTM的改良版，LSTM引入了很多内容，导致参数变多，也使得训练难度加大了很多。因此很多时候我们往往会使用效果和LSTM相当但参数更少的GRU来构建大训练量的模型。



# Transfer Learning

遷移式學習

把B領域中的知識遷移到A領域中來，提高A領域分類效果 :

- 舉例 : 當你想辨識馬兒，但能用的訓練樣本卻有限，你可以用其他動物辨識的訓練模型，加以調整，來訓練出辨識馬兒的模型。

優點 :

1. 降低訓練時特徵提取時間

    - 因為你已經拿了別人標記好的資料

2. 降低淺層網絡訓練時間

    - 前面幾層 (淺層) 通常是拿來判斷最基本的圖形，例如橫線、直線、非常小的圓圈、...，這代表前面幾層 layer 跟要判斷的目標沒有最直接關係，卻也極為重要，因為不管是甚麼圖形，都是由像素級的圖形拼湊而成。
      因此一般會將別人模型的前面幾層 layer 保留，僅換掉後面幾層 layer。

2. **避免訓練資料太少造成的過擬和**

缺點 :

1. 訓練準確率得不到保證


參考資料 :

1. [Machine Learning — Transfer Learning (遷移學習)](https://medium.com/@yuhsienyeh/machine-learning-transfer-learning-%E9%81%B7%E7%A7%BB%E5%AD%B8%E7%BF%92-5095f8a14367)

    - Zero-shot learning : 最廣為人之的應用就是語言翻譯

      - 訓練時 : 先對**英日互翻** **英韓互翻**，會產生一些英、日、韓的向量
        測試時 : 要做**日韓互翻**，但此時系統完全不知道日文「私は」是韓文的什麼，(因為一開始只有訓練英日 英韓互翻)，但是將「私は」向量化之後，我們就可以在語言向量空間這個“資料庫”**查詢最接近「私は」的韓文向量是哪一句**，來達到日韓互翻的目的

2. [深度學習模型-遷移學習(Transfer Learning) 概述@csdn:lqfarmer](https://bigdatafinance.tw/index.php/tech/methodology/988-transfer-learning)

3. [Day01 Transfer Learning 遷移式學習@gueiyajhang](https://tw.coderbridge.com/series/d4b5a1a1565e4e7a9cd14618ffe6146f/posts/54584ea6d4c240aeb3b8ae4af3a0531a) Feb 22, 2020

# AutoEncoder

自動編碼器

其中一個概念 :

![csdn自动编码器AutoEncoder学习总结@mpk_no1](https://img-blog.csdn.net/20170723202257957?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbXBrX25vMQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
圖片來源:https://img-blog.csdn.net/20170723202257957?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbXBrX25vMQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center

input 輸入到 encoder 中，作一些處理，會得到一組 code，將這組 code 輸入到 decoder 中，得到一個輸出，

**如果这个输出与输入input很像的话**，那我们就可以相信这个中间向量code跟输入是存在某种关系的，也就是存在某种映射，**那么这个中间向量code就可以作为输入的一个特征向量**。

- 通过调整encoder和decoder的参数，使得输入和最后的输出之间的误差最小。
<!--版权声明：本文为CSDN博主「mpk_no1」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/mpk_no1/article/details/75201582-->

便找到可以代表输入数据的最重要的因素，**找到可以代表原信息** ( input ) **的主要成分**。


參考資料 :

1. [重新認識 AutoEncoder，以及訓練 Stacked AutoEncoder 的小技巧@弱弱開發女子](https://medium.com/%E5%BC%B1%E5%BC%B1%E9%96%8B%E7%99%BC%E5%A5%B3%E5%AD%90-%E5%9C%A8%E6%9D%B1%E4%BA%AC%E7%9A%84%E9%96%8B%E7%99%BC%E8%80%85%E4%BA%BA%E7%94%9F/autoencoder-%E6%88%91%E5%B0%8D%E4%B8%8D%E8%B5%B7%E4%BD%A0%E4%B9%8B-%E9%87%8D%E6%96%B0%E8%AA%8D%E8%AD%98autoencoder-%E7%AC%AC%E4%B8%80%E7%AF%87-d970d1ad9971) Jun 13, 2018

    - 一些基本概念

2. [自动编码器AutoEncoder学习总结@mpk_no1](https://blog.csdn.net/mpk_no1/article/details/75201582) 2017-07-23

# GNN

生成對抗網路

gnn 演算法操作的過程 : [GAN Lecture 1 (2018): Introduction](https://www.youtube.com/watch?v=DQNNMiAP5lw&list=PLJV_el3uVTsMq6JEFPW35BCiOQTsoqwNw) 15:13 + 21:20


cycle gan :

參考資料 :

1. [GAN Lecture 1 (2018): Introduction@李宏毅](https://www.youtube.com/watch?v=DQNNMiAP5lw&list=PLJV_el3uVTsMq6JEFPW35BCiOQTsoqwNw)

    - gnn 操作的過程 : 15:13 + 21:20


---

### 對資料的正規化

(資料前處理使用 (而已?))


---

# NLP


## 資料形式說明

舉例 :

| id| 文本一     | 文本二    | label     |
| :--------| :------------- | :------------- |:------------- |
| 0| 一個中文詞彙    | 這是漢語的單字   |related    |
| 1| 我看見一隻貓    | 吐司草莓        |unrelated    |
|...| ...            | ...            | ...         |
| n| 天空是廣大的    | 浩瀚 蒼穹       | related     |

**label** 是我自己對兩個文本的標記，要讓機器去學習的東西


## 數據前處理

將文字轉成數字，方便電腦處理。

步驟 :

1. 文本轉換

    ```
    一個中文詞彙 --> 一個 / 中文 / 詞彙
    ```

2. 建立字典並將文本轉成數字序列

    ```
    建立字典 : {'一個': 1, '中文': 2, '詞彙': 3}
    ```
    ```
    將文本轉成數字序列 : ['一個', '中文', '詞彙'] --> [1, 2, 3]
    ```

3. 序列的 Zero Padding

    - 讓所有序列的長度一致，方便之後的 NLP 模型處理

4. 將正解做 One-hot Encoding

    1. 將label變成數值

    2. 再將各資料對應的label轉成向量

    3. 最終會變成像這樣 :
        | id| 文本一     | 文本二    | label     |
        | :--------| :------------- | :------------- |:------------- |
        | 0| 一個中文詞彙    | 這是漢語的單字   |`[0., 1., 0.]`    |

        `[0., 1., 0.]` 代表文本一與文本二 為 `related` 的**機率**為 100%
        `[0.7, 0.2, 0.1]` 代表文本一與文本二 有70%機率為 `unrelated`、 20%機率為`related`、10%機率為`unknown`

5. 將整個資料集拆成訓練資料集 & 驗證資料集

    - 以方便之後測試模型的效能。

### 1. 文本轉換

文本轉換( Text segmentation )

將一連串文字 (文本) 切割成多個有意義的單位，單位可以是

- 一個中文漢字 / 英文字母（Character）
- 一個中文詞彙 / 英文單字（Word Segmentation）

    ```
    一個中文詞彙 -->
    一個 / 中文 / 詞彙
    ```

- 一個中文句子 / 英文句子（Sentence）

切完之後的每個文字片段在 NLP 領域裡頭習慣上會被稱之為 **Token**

#### 1-1 Word Segmentation

英文 :

- 按照空格分割，分出各個單字

中文

  1. 藉助 [<font color =LightSkyBlue>Jieba</font>](https://github.com/fxsjy/jieba) 這個中文斷詞工具

### 2. 建立字典並將文本轉成數字序列

建立字典 :

```
一個中文詞彙 -->
{'一個': 1, '中文': 2, '詞彙': 3}
```

將文本轉成數字序列 :

```
['一個', '中文', '詞彙'] -->
[1, 2, 3]
```

應用 :

```
中文詞彙一個 -->
['中文', '詞彙', '一個'] -->
[2, 3, 1]
```

#### 使用Keras

- 建立字典過程會太繁瑣 ( 要一個個比對哪個字沒有出現過，然後加到字典裡 )，所以全程用 **<font color =LightSkyBlue>Keras</font>** 來做

1. 建立字典

    ```py
    import keras
    MAX_NUM_WORDS = 10000
    # 限制字典只能包含 10,000 個詞彙，一旦字典達到這個大小以後，剩餘的新詞彙都會被視為 Unknown
    # 避免字典過於龐大

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
    # .Tokenizer會將一段文字轉換成一系列的詞彙（Tokens），並為其建立字典。
    ```

2. 利用生成的字典，將文本轉成數字序列

### 3. 序列的 Zero Padding

讓所有序列的長度一致，方便之後的 NLP 模型處理

#### 使用Keras

```python
keras.preprocessing.sequence.pad_sequences(XXX,maxlen=4) # 長度設為4
# 長度超過4的序列尾巴會被刪掉；
# 原來長度不足的序列，我們則會在詞彙前面補零
```

### 4. 將正解做 One-hot Encoding

1. 將 label 變成數值

    ```python
    import numpy as np

    # 定義每一個分類對應到的索引數字
    label_to_index = {
        'unrelated': 0,
        'related': 1,
        'unknown': 2
    }

    # 將分類標籤對應到剛定義的數字
    y_train = train.label.apply(lambda x: label_to_index[x])
    # lambda 解釋 : [Python- lambda 函式@Sean Yeh](https://medium.com/seaniap/python-lambda-%E5%87%BD%E5%BC%8F-7e86a56f1996)
    # lambda簡單來說，就是 傳入值為x，傳回值為 label_to_index[x]
    # .apply 解釋 : [Pandas教程 | 数据处理三板斧——map、apply、applymap详解](https://zhuanlan.zhihu.com/p/100064394)
    # apply簡單來說，就是把 label 的值轉成 label_to_index，比如 related --> 1

    y_train = np.asarray(y_train).astype('float32')
    # asarray 解釋 : [np.array()和np.asarray()的区别](https://www.cnblogs.com/Renyi-Fan/p/13773546.html#_label1)
    # asarray都可以将结构数据转化为ndarray
    # nparray 解釋 : [[Day14]Numpy的ndarray！](https://ithelp.ithome.com.tw/articles/10195434)
    # ndarray簡單來說是一個快速的且可以節省空間的多維度陣列，提供向量運算以及複雜的功能
    ```

2. 將各資料 label 做 one-hot encoding，也就是將各資料的 label 向量化

    ```python
    # 使用keras
    y_train = keras.utils.to_categorical(y_train)
    ```

    | id| 文本一     | 文本二    | :heavy_check_mark:label     |
    | :--------| :---| :---|:----|
    | 0| 一個中文詞彙    | 這是漢語的單字   |[0., 1., 0.]|
    | 1| 我看見一隻貓    | 吐司草莓     |[1., 0., 0.]   |
    |...| ...        | ...      | ...         |
    | n| 天空是廣大的    | 浩瀚 蒼穹  |[0., 1., 0.]    |

    也就是把 label 的每種值變成一個維度，
    就可以想成 :
    - `[0., 1., 0.]` 代表文本一與文本二 為 `related` 的**機率**為 100%

    - `[0.7, 0.2, 0.1]` 代表文本一與文本二 有70%機率為 `unrelated`、 20%機率為`related`、10%機率為`unknown`


### 5. 將整個資料集拆成訓練資料集 & 驗證資料集

一般來說，我們在訓練時只會讓模型看到訓練資料集（Training Set），並用模型沒看過的驗證資料集（Validation Set）來測試並調整模型的參數，

接著用與整個資料集完全獨立的測試資料集（Test Set）來展示模型的最終成果


#### 使用[scikit-learn](https://scikit-learn.org/stable/index.html)

```python
from sklearn.model_selection import train_test_split

# 小彩蛋
RANDOM_STATE = 9527

x1_train, x1_val, \
x2_train, x2_val, \
y_train, y_val = \
    train_test_split(
        x1_train, x2_train, y_train,
        # 文本一,文本二,【label】 都分成 訓練資料集 與 驗證資料集
        # 將x1_train 分成 訓練資料集 x1_train + 驗證資料集 x1_val，其他也是

        test_size= 0.1, # 分成 10% 驗證資料集 + 90% 訓練資料集
        random_state = RANDOM_STATE # 亂數種子吧?
)
```


References:

1. [進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南@leemeng](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#3-%E9%96%80%E6%8E%A8%E8%96%A6%E7%9A%84%E7%B7%9A%E4%B8%8A%E8%AA%B2%E7%A8%8B) 2018-12-24;20221224


## Word Embedding

詞嵌入，就是把詞變成向量

我們的資料形式原本長這樣:

| id| 文本一     | 文本二    | label     |
| :--------| :------------- | :------------- |:------------- |
| 0| 一個中文詞彙   | 這是漢語的單字   |	related    |

經過上面的數據預處理，資料變成這樣 :

| id| 文本一     | 文本二    | label     |
| :--------| :------------- | :------------- |:------------- |
| 0| [0, 1, 2, 3]    | [4, 5, 6, 7]   |`[0., 1., 0.]`    |

然而 一個字/一個詞 變成一個數字，並不能從該數字看出其原義

所以需要將每 一個字/一個詞 變成向量(或稱張量:**Tensor**)，才能用數學的角度處理語意 ( 此技術稱為 word embedding )

但是詞向量的維度也不能毫無意義，比如下圖左邊 :

![](./mdPic/NLP_wordEmbedding01.png)

看不出為甚麼雞在最右邊，為甚麼豬跟魚的y值如此接近

而上圖右邊則看得出來，x軸越往右代表水生動物，越往左代表陸生動物 ; y軸則是腳的數量。

**我們可以透過平常訓練神經網路的反向傳播算法（Backpropagation）來找到最適合每個詞的詞向量**


### 1. 做 word embedding

#### 使用Keras

我們可以使用Keras 的 `Embedding layer` 來幫我們把詞轉換成詞向量

```python
from keras import layers
embedding_layer = layers.Embedding(MAX_NUM_WORDS, 3)
# .Embedding( 我們字典的大小, 詞向量的維度 )
# 詞向量的維度通常設 128、256 或甚至 1,024。
# 筆記為了方便，詞向量的維度設 2
```

我們的資料經過word embedding後變成這種形式 :

| id| 文本一     | 文本二    | label     |
| :--------| :------------- | :------------- |:------------- |
| 0| [ `[0.212, 0, 0.788]`,</br> `[0.2, 0.7, 0.1]`,</br>`[0.212, 0.1, 0.688]`,</br> `[0.5, 0.3, 0.2]` ]    | [ `[0.163, 0.93, 0.58]`,</br>`[0.3, 0.2, 0.5]`,</br>`[0.21, 0.11, 0.68]`,</br>`[0.528, 0.344, 0.452]` ]   |`[0., 1., 0.]`    |

### 2. 接著把詞向量丟入RNN / LSTM 裏頭，讓模型逐步修正隨機初始化的詞向量，使得詞向量裡頭的值越來越有意義


後面關於神經網路的步驟 : https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#3-%E9%96%80%E6%8E%A8%E8%96%A6%E7%9A%84%E7%B7%9A%E4%B8%8A%E8%AA%B2%E7%A8%8B :
![](https://leemeng.tw/images/nlp-kaggle-intro/siamese-network.jpg)
圖片來源 :https://leemeng.tw/images/nlp-kaggle-intro/siamese-network.jpg



References:

1. [進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南@leemeng](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#3-%E9%96%80%E6%8E%A8%E8%96%A6%E7%9A%84%E7%B7%9A%E4%B8%8A%E8%AA%B2%E7%A8%8B)
## NLP實做筆記 :

### pyrhon tensorflow

```python
# tensorflow 隨機抽樣使用  #tensorflow v2.7
predicted_id = tf.random.categorical(predictions, num_samples=1)
predicted_id.numpy()
# numpy不能接在tensorflow後面
# 用 np.max(predicted_id.numpy())，取得隨機抽樣機率最高的值
```

### colab code紀錄

```python
#小說共有 5387719 中文字，才能跑出我們想要的樣子
!pip install tf-nightly-gpu #如果有 GPU 則強烈建議安裝 GPU 版本的 TF Nightly，訓練速度跟 CPU 版本可以差到 10 倍以上。
!pip install tensorflow==2.7
!pip install google.colab #如未安裝取消註解後執行
import numpy as np
import os
from google.colab import drive
import tensorflow as tf
drive.mount('/content/drive')
##出現提示欄進行授權
os.chdir('/content/drive/My Drive/homework/AI/') #切換該目錄
os.listdir() #確認目錄內容
# https://ithelp.ithome.com.tw/articles/10234373

path = 'novel_full.txt'
f = open(path, 'r')
text = f.read()
print(text[9505:9702])
f.close()
n = len(text)
w = len(set(text))
print(f"小說共有 {n} 中文字")    # 小說共有 1535268 中文字
print(f"包含了 {w} 個獨一無二的字") #包含了 3513 個獨一無二的字



dic_num = w
#import tensorflow as tf
# 初始化一個以字為單位的 Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=dic_num,  # 限制字典只能包含 3513 個詞彙，設剛好3513真的對嗎?
        char_level=True,
        filters=''
)

# 讓 tokenizer 讀過全文，將每個新出現的字加入字典並將中文字轉成對應的數字索引
tokenizer.fit_on_texts(text)
text_as_int = tokenizer.texts_to_sequences([text])[0]


# 方便說明，實際上我們會用更大的值來讓模型從更長的序列預測下個中文字
SEQ_LENGTH = 100  # 數字序列長度
BATCH_SIZE = 128 # 幾筆成對輸入/輸出

# text_as_int 是一個 python list
# 我們利用 from_tensor_slices 將其轉變成 TensorFlow 最愛的 Tensor
characters = tf.data.Dataset.from_tensor_slices(text_as_int) # 把每個字從數值轉成tensor

# 將被以數字序列表示的文本拆成多個長度為 10 的序列
# 並將最後長度不滿 SEQ_LENGTH 的序列捨去
sequences = characters.batch(SEQ_LENGTH + 1,drop_remainder=True)

# 全文所包含的成對輸入/輸出的數量
steps_per_epoch = len(text_as_int) // SEQ_LENGTH # 將數字序列以10為長度切分

# 這個函式專門負責把一個序列
# 拆成兩個序列，分別代表輸入與輸出
# （下段有 vis 解釋這在做什麼）
def build_seq_pairs(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# 將每個從文本擷取出來的序列套用上面定義的函式，拆成兩個數字序列作為輸入／輸出序列
# 再將得到的所有數據隨機打亂順序
# 最後再一次拿出 BATCH_SIZE（128）筆數據，作為模型每一次訓練步驟的所需使用的資料量
ds = sequences.map(build_seq_pairs)\
    .shuffle(steps_per_epoch)\
    .batch(BATCH_SIZE,drop_remainder=True)



# 超參數
EMBEDDING_DIM = 512
RNN_UNITS = 1024

# 使用 keras 建立一個非常簡單的 LSTM 模型
model = tf.keras.Sequential()

# 詞嵌入層
# 將每個索引數字對應到一個高維空間的向量
model.add(
    tf.keras.layers.Embedding(
        input_dim=dic_num,
        output_dim=EMBEDDING_DIM,
        batch_input_shape=[
            BATCH_SIZE, None]
))

# LSTM 層
# 負責將序列數據依序讀入並做處理
model.add(
    tf.keras.layers.LSTM(
    units=RNN_UNITS,
    return_sequences=True,
    stateful=True,
    recurrent_initializer='glorot_uniform'
))

# 全連接層
# 負責 model 每個中文字出現的可能性
model.add(
    tf.keras.layers.Dense(dic_num) # 因為包含了 3513 個獨一無二的字、吐出3513 維的 Tensor
    )

model.summary()



# 超參數，決定模型一次要更新的步伐有多大
LEARNING_RATE = 0.001
#https://cloud.tencent.com/developer/article/1725772

# 定義模型預測結果跟正確解答之間的差異
# 因為全連接層沒使用 activation func
# from_logits= True
def loss(y_true, y_pred): # loss function : 交叉熵
    return tf.keras.losses\
    .sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True)

# 編譯模型，使用 Adam Optimizer 來最小化
# 剛剛定義的損失函數
model.compile(
    optimizer=tf.keras.optimizers.Adam( # 設定優化器
        learning_rate=LEARNING_RATE),
    loss=loss # 設定loss function
)



EPOCHS = 30



%load_ext tensorboard
%reload_ext tensorboard
callbacks = [
    tf.keras.callbacks\
        .TensorBoard("logs"),
    # 你可以加入其他 callbacks 如
    # ModelCheckpoint,
    # EarlyStopping
]
# ds : 前面使用 tf.data 建構的資料集
history = model.fit(
    ds,
    epochs=EPOCHS,
    callbacks=callbacks
)



model.save_weights('/content/drive/My Drive/homework/AI/a.ckpt') # checkpoint檔案




# 預測
import tensorflow as tf

# 跟訓練時一樣的超參數，
# 只差在 BATCH_SIZE 為 1
EMBEDDING_DIM = 512
RNN_UNITS = 1024
BATCH_SIZE = 1

# 專門用來做生成的模型
infer_model = tf.keras.Sequential()

# 詞嵌入層
infer_model.add(
    tf.keras.layers.Embedding(
        input_dim=dic_num,
        output_dim=EMBEDDING_DIM,
        batch_input_shape=[
            BATCH_SIZE, None]
))

# LSTM 層
infer_model.add(
    tf.keras.layers.LSTM(
    units=RNN_UNITS,
    return_sequences=True,
    stateful=True
))

# 全連接層
infer_model.add(
    tf.keras.layers.Dense( # 出來一個dic_num維的字
        dic_num))

# 讀入之前訓練時儲存下來的權重
infer_model.load_weights('/content/drive/My Drive/homework/AI/a.ckpt') # 檔案位置，簡單的說它儲存了變量(variable) 的名字和對應的張量(tensor) 數值
infer_model.build(
    tf.TensorShape([1, None]))





#text_generated = '就在此时，他' # 這是我一開始的文本，接著要繼續生成
x=310
text_generated = tokenizer.index_word[x]
# 代表「喬」的索引
seed_indices = [x] ### 先隨機挑一個
print(tokenizer.index_word[x])

for i in range(500):

  #seed_indices = tokenizer.texts_to_sequences([text_generated])[0] # 把之前擁有的文本全部變成數字序列


  # 增加 batch 維度丟入模型取得預測結果後
  # 再度降維，拿掉 batch 維度
  input = tf.expand_dims(seed_indices, axis=0) # 當作沒看見好了 # axis=0表示在原有的张量的第一维扩充 # https://blog.csdn.net/hgnuxc_1993/article/details/116941367
  predictions = infer_model(input)  # 因為batch = 1，所以input會一個字一個字讀入 # prediction會是輸入全部字後的下一個字 (以3513維表示，仍是機率，還未成形)
  predictions = tf.squeeze(predictions, 0) # 當作沒看見好了 # 给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。
  # 所以prediction還是在大約3513維機率的狀態

  # 利用生成溫度影響抽樣結果
  temperature = 1
  predictions /= temperature # 影響機率

  # prediction 成形 成 sampled_indices
  sampled_indices = tf.random.categorical(predictions, num_samples=1)
  # 把prediction的3513維隨機抽樣其中 1 個，(3513維都看做是機率)，禿常會抽到機率最大的那維，
  # 會回傳納維的index，就可以對照字典index得到是哪個字
  # https://blog.csdn.net/menghuanshen/article/details/105356239

  input_eval = tf.expand_dims([sampled_indices], 0) # 不知道幹嘛 # 就prediction的3513維

  # 對照字典
  #sampled_indices = tf.squeeze(sampled_indices,axis=-1)
  #print(tokenizer.sequences_to_texts(sampled_indices.numpy()))
  #partial_texts = tokenizer.index_word[sampled_indices]
  #print(sampled_indices.numpy())
  maxindex = np.max(sampled_indices.numpy())
  partial_texts = tokenizer.index_word[maxindex]

  #if(text_generated[-1] != partial_texts[0]):
      #text_generated += partial_texts[0] # 每次生成的東西壘加
  text_generated += partial_texts[0] # 每次生成的東西壘加
  seed_indices = [maxindex]
  #print(maxindex)

print(text_generated)
#print(text_generated.split('\n')[0].split('。')[0]) # 結果，會有50+個字

```

---

# yolo


1. :star: 資料夾的創建方式有特定規則，建議不要隨意放置

    - [YOLOv5 實現目標檢測（訓練自己的資料集實現貓貓識別）](https://tw511.com/a/01/29504.html)

    ```
    ├── yolo
        ├── data
        │   ├── Annotations  進行 detection 任務時的標籤檔案，xml 形式，檔名與圖片名一一對應
        │   ├── images  存放 .jpg 格式的圖片檔案
        │   ├── ImageSets  存放的是分類和檢測的資料集分割檔案，包含train.txt, val.txt,trainval.txt,test.txt
        │   ├── labels  存放label標註資訊的txt檔案，與圖片一一對應


        ├── ImageSets(train，val，test建議按照8：1：1比例劃分)
        │   ├── train.txt  寫著用於訓練的圖片名稱
        │   ├── val.txt  寫著用於驗證的圖片名稱
        │   ├── trainval.txt  train與val的合集
        │   ├── test.txt  寫著用於測試的圖片名稱
    ```

### yaml

##### yolov5 :

```yaml
# 設定圖檔路徑
path: ../datasets/eggs  # 資料根目錄
train: images/train     # 訓練用資料集（相對於 path）
val: images/train       # 驗證用資料集（相對於 path）
test:                   # 測試用資料集（相對於 path，可省略）

# 物件類別設定
nc: 2           # 類別數量
names: ['egg','frog']  # 類別名稱

# code來源:https://officeguide.cc/pytorch-yolo-v5-object-egg-detection-models-tutorial-examples/
```

### Yolo Loss

訓練完成後會顯示的指標

Yolov5中的三個loss指標分別為cls_loss、box_loss與obj_loss
- cls_loss : 用來判斷模型是否能夠準確**分**出目標屬於哪**類**，如貓狗等等…。
- box_loss : 則是模型回歸出的x, y, w, h座標點數值。
- obj_loss : 則是除了考慮座標數值外還要考慮與目標物的iou**面積**，來確保整體模型框的位置是否準確。

來源 : [YOLO-V5 使用教學@蔡承翰](https://hackmd.io/@4XoSjtMaS46Zzn7DwmEIEQ/SkC0ceHlF#%E6%A8%A1%E5%9E%8B%E8%A8%93%E7%B7%B4)

---

# 機器學習任務

分 regression、classification、cluster

### regression

像是回歸線、(找目標資料所在定位那種)

### classification

Q : 若有十類(已知)，請問目標屬於哪一類

### cluster

Q : 請幫我將這些資料分成十類


---

# 實作概念

怎麼決定模型大小(ex:使用層數) : 經驗法則，熟的人就知道一開始大概要用多大，否則就一直去試
