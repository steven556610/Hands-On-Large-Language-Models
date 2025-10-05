[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
[Should We Still Pretrain Encoders with Masked Language Modeling?](https://arxiv.org/html/2507.00994v1)

* [自回归语言模型与大语言模型（CLM和LLM）的区别](https://blog.csdn.net/qq_42755230/article/details/142848849)

* [大型語言模型的預訓練任務](https://medium.com/@albertchen3389/%E5%A4%A7%E8%AA%9E%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%A0%90%E8%A8%93%E7%B7%B4%E4%BB%BB%E5%8B%99-b831dcf8f6f7)
CLM 因果語言建模，像是GPT。CLM 的模型與MLM 相比，在捕捉上下文方面的效果可能沒有那麼好。CLM 模型是一個自回歸模型，單向，從左到右的架構，並且只處理前面的token。一次預測一個token。專注在文本生成的任務。

BERT，採用無監督式預訓練＋監督微調，預訓練任務與GPT不同，主要是Masked language modeling(MLM) 和 Ｎext Sentense Prediction(NSP)

* [【Machine Learning】：LLM的一些學習綜整](https://ithelp.ithome.com.tw/articles/10313019)
    1. Autoregressive v.s Non-autoregressive
    這個在李鴻毅2021的機器學習課程就有提到過，Autoregressive以及Non-autoregressive重點差別在於 輸出產生方式 (補充：輸出可以是文字, phoneme)：

    Autoregressive (AT)：輸出從 BEGIN 的 Token 開始，將上一個文字的預測輸出當作是下一個預測文字的輸入，直到輸出END才結束句子生成。
    Non-autoregressive (NAT)：而是一次吃的是一整排的 BEGIN 的 Token，把整個句子一次性都產生出來
    有關於NAT 之 控制BEGIN的個數可以用以下方法實現：
    1. 另外訓練一個 Classifier： 吃 Encoder 的 Input，輸出是一個數字，代表 Decoder 應該要輸出的長度。
    2. 給它一堆 BEGIN 的 Token，直到輸出END才結束句子生成。
    3. NSP 以及 MLM 是甚麼？
    NSP跟MLM都是pre-train model的常用訓練方法，我最初接觸的時候是在閱讀BERT用於pre-trained的時候，使得文字語料可以對模型進行預訓練：

    MLM 又被稱作是 Masked Language Modeling
    簡而言之，該任務會用符號(ex:"[MASK]") 隨機掩蓋掉輸入的Token，

    順帶一提，在BERT的原文是這樣說的：

    In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random.

    而NSP的缺點會導致pre-training跟fine-tuning兩階段訓練目標差距過大的問題，這點也在BERT的論文也有被提到：

    Although this allows us to obtain a bidirecetional pre-trained model, a downside is that we are creating a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning.

    因此BERT也同時採用了NSP的方法。

    NSP 又被稱作是 Next Sentense Prediction
    該訓練的任務目標是建立模型理解詞彙間的能力，例如「小明愛小華」「小華愛小明」這兩句雖然人名對調，但意思完全不一樣對吧？

    NSP 在訓練上會採用二分類的方法，也就是隨機找文章中的上下兩段話，去對機器進行上下文的QA問答。

    舉例來說：我們會拿一個句子去問機器說「小明愛小華」的下一句是不是「但小華有男朋友了」？然後機器就會根據當下訓練的參數去進行IsNext或是NotNext的猜測，這時我們只要把答案給機器看(答案的形式為IsNext/NoNext)，告訴機器下一句其實是「但小華只有8歲」，答案是NoNext，這樣就可以完成訓練摟~

[Understanding Causal LLM’s, Masked LLM’s, and Seq2Seq: A Guide to Language Model Training Approaches](https://medium.com/@tom_21755/understanding-causal-llms-masked-llm-s-and-seq2seq-a-guide-to-language-model-training-d4457bbd07fa)