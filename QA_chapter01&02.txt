env build
    macOS:
        package: 
        > torch version downgrade
        > bitsandbytes downgrade
        sudo xcodebuild -license

jupyter 1:
    https://ithelp.ithome.com.tw/m/articles/10303625
    adjust cuda to cpu
    讀入模型，並且用模型去生成了一些文字
    會有model 和 tokenizer，使用 transformers 的 AutoModelForCausalLM, AutoTokenizer
    裡面的參數設定，有修改'cpu'。後續應該是可以調整裡面的模型。
    causalLM 是一種，還有沒有其他的 module。
    後續會建立一個generator，使用 transformers 的 pipeline，需要理解裡面的參數設定。

    message 是一個dictionary。
    裡面有放role 和 content 作為key，
    value 一個是user 一個是 要給模型的文字。

jupyter 2:
1. downloading and running an llm
    > 這邊使用AutoCasualLM
    先讀入model 和 tokenizer。
    prompts 需要先使用tokenizer 進行轉換，轉成ids。
    gneneration output 會使用到model.generate(input_ids=input_ids, max_new_tokens=20)
    
    tokernizer.decoder(generation_output[0]) #output: Dear
    input_ids 會是tensor([[a,b,c]]) #a,b,c 會是數字
    decode過後會變成文字
    generation_output本身也是 tensor([[a,b,c]])
    逐步使用tokenizer.decode 會變成文字。並且有些文字本身是分開來的

2. computing trained llm tokenizers
    跑各種不同的模型還有設定顏色，基本上各大科技公司都有模型。
    在這一個段落，使用不同模型的tokenizer 來看文字進去tokenizer過後，會產生什麼變化
    再使用tokenizer.decode來分析output
    使用show_tokens 來設定文字和顏色

3. Contextualized Word Embeddings From a Language Model (Like BERT)
    > 這邊使用的AutoModel 和AutoTokenizer
    tokenizer 讀入分詞器
    model 讀入模型
    token 則是文字經過分詞器的結果，這是一個包含多個元素的dictionary。
    如果要tokenizer.decode 要使用 tokens['input_ids']
    output 是模型的主要輸出，會是一個pytorch tensor。代表了每個輸入token在模型最後一層的向量表示。
    這一個向量包含了該token在鉅子上下文中的語意資訊。
    hello world 最後產生四個token，然後在output就變成了toch.size([1,4,384])
    1 代表batch size 因為我們只輸入了一個句子
    4 代表分詞器的分詞數量。
    384 就是模型最後一層的hidden size 大小。

    important explain:
        使用 **token 進模型，是一個distionary unpakcing 的作法。
        tokens本身是一個dictionary。使用**token 可以讓模型參數直接進行對應。
        變成model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        讓字典的鍵值對，可以展開來作為模型的參數引入。
        benefit: 保持程式碼簡潔。

        為什麼 output是取 model(**tokens)[0]，因為其實模型的output會包含許多不同的tuple 或是ModelOutput object。
        這一個tuple裡面的不同位置，存放著不同的輸出資訊。
        位置[0] 裡面多是存放，最後一個的隱藏狀態(Last Hidden State)，代表模型對輸入的最終理解。
        位置[1] 可能是池化輸入（Pooler Output）或是分類頭的輸出（Classification Head）。
        這個部分取決於使用什麼樣的模型。

4. Text Embeddings (for sentenses and whole documents)
    > 使用 sentence_transformers
    讀入模型後。
    要把文字轉換成向量，使用model.encode('')
    就可以把文字轉換成向量

5. word embedding beyond LLMs
    > 使用 gensim.downloader as api
    下載模型 model = api.load("glove-wiki-gigaword-50")
    並且使用到model.most_similar(model['king'], topn=11)
    就可以列出與king這一個字最接近的11個詞

6. Recommending songs by Embeddings
    讀入playlist 還有songs_df。
    * NOTE: playlists是 list 裡面包 list of str
    把playlist 放進去訓練的假設，是把每一個歌曲都想像成詞彙，而一個playlist就是一句想要表達的話。
    playlist 裡面應該都是比較相近的歌。如果兩個歌曲ID 經常出現在同一個播放清單中。
    在經過word2vec 的過程，就會認為他們兩個的意義是類似的。
    ＊ 意思就是playlist 就是分群後的結果，把分群後的結果來轉換為向量。
    這一個概念很有趣。抽象的用歌曲id來表示成文字，然後再轉換成數字特徵。
    針對words embed 後的結果，會是維度32的向量。
    最後有寫一個function來抓取對應的歌曲列表的內容。
    * 有使用 pip install -upgrade scipy gensim。
    因為原來版本在word2vec 會遇到error。

application idea:

* 讀入LLM 可以拿來做哪些事情？ 分析文本，製作回覆機器人？
* 推薦系統也是一個選項，並且在這邊的範例，給了一個很不一樣的思路。
