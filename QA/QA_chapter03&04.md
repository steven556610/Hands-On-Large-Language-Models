* jupyter 03:
    # NOTE: 讀入transformer的模型時，如果沒有gpu 就要記得在device_map 換成“cpu"。
    主要來嘗試文字生成。
    就會需要先讀入一個LLM 模型。
    目前使用AutoModelForCasualLM，也有讀入tokenzier，pipeline
    使用pipeline包裝成一個生成器(generator)。
    
    使用方式就會變成
    prompt 裡面放string
    output = generator(prompt) #一個dictionary 裡面只有一給key ('generated_text')和value。
    print(output[0]['generated_text'])

    print(model) #model 本身是一個已經預訓練好的模型。
    1. 總體結構：Phi3ForCausalLM
        這個名稱表示它是用於 因果語言建模 (Causal Language Modeling, CausalLM) 的模型。

        用途： 預測序列中的下一個詞元 (token)，因此非常適合文本生成（如對話、寫作、程式碼補全）。

        它包含兩個主要組成部分：

        model (Phi3Model)： 模型的主體，負責處理輸入並產生最終的隱藏狀態 (hidden states)。

        lm_head (Linear)： 語言模型頭部，負責將隱藏狀態轉換為詞彙表上的機率分佈。
    2. 
        A.
        |組件|參數|數值|意義|
        |:----:|:----:|:----:|:----:|
        |embed_tokens|Embedding(32064, 3072, ...)|32064|詞彙表大小 (Vocabulary Size)。這是模型能理解的不同詞元數量|
        |embed_tokens|Embedding(32064, 3072, ...)|3072|隱藏層維度 (Hidden Size)。這是每個詞元被轉換成的向量維度。|
        |lm_head|Linear(..., out_features=32064, ...)|32064	|輸出維度與詞彙表大小一致，用於預測下一個詞元。|
        |padding_idx|32000|指定填充詞元的索引，用於處理不同長度的輸入序列。|
        
        B.
        |組建|參數|數值|意義|
        |layers|ModuleList(0-31)|32|層數 (Number of Layers)。模型包含 32 個相同的解碼器層，這是模型學習深度的關鍵。|
    
    3. 核心模塊：Phi3DecoderLayer (共 32 層)

        每個解碼器層是 Transformer 架構的基礎單元，包含自我注意力 (Self-Attention) 和多層感知器 (MLP) 兩大模塊。

        A. 注意力機制 (self_attn: Phi3Attention)
        投影層 (qkv_proj)： Linear(in_features=3072, out_features=9216)。

        輸入 3072，輸出 9216 (3072×3)。這表示它將 Q (Query)、K (Key)、V (Value) 的投影計算合併為一步。

        輸出投影 (o_proj)： Linear(in_features=3072, out_features=3072)。

        旋轉位置嵌入 (rotary_emb): 使用 RoPE (Rotary Position Embedding)，這是一種常見於現代 LLM 的高效位置編碼方式，取代了傳統的絕對位置編碼。

        B. 前饋網路 (mlp: Phi3MLP)
        門控和提升投影 (gate_up_proj)： Linear(in_features=3072, out_features=16384)。

        這暗示使用了 Gated MLP 結構，例如 SwiGLU (Swish-Gated Linear Unit) 或類似結構，它將輸入維度 3072 擴展到一個更大的中間維度。

        激活函數 (activation_fn): 使用 SiLU (Sigmoid Linear Unit)，一種平滑的非線性激活函數，常用於高效能的 LLM 中。

        降維投影 (down_proj)： Linear(in_features=8192, out_features=3072)。

        這層將擴展後的維度 8192（很可能由 16384 分割或處理而來）重新映射回模型的隱藏維度 3072。

        C. 標準化與正則化 (Normalization and Regularization)
        標準化 (input_layernorm, post_attention_layernorm, norm): 全部使用 Phi3RMSNorm (Root Mean Square Normalization)。

        RMSNorm 比傳統的 LayerNorm 具有更高的計算效率，是許多高性能 Transformer 模型（如 Llama、Mistral）的首選。

        Dropout： 所有的 Dropout 參數 p=0.0。

        這意味著模型在推理階段運行，或者在預訓練階段未使用 Dropout（這對於大型模型來說並不罕見，因為它們通常使用其他正則化技術，或在微調時才加入 Dropout）。

        🚀 總結與主要設計理念
        這個 Phi-3 Small 模型體現了現代高效能輕量級 LLM 的幾個關鍵設計選擇：

        Decoder-Only 架構： 專注於因果語言建模，擅長生成。

        RoPE 和 RMSNorm： 採用了 Llama 家族和 Mistral 成功使用的高效能組件，以提高速度和穩定性。

        精簡的維度 (3072 Hidden Size, 32 層)： 相較於數萬億參數的模型，Phi-3 Small 旨在以更少的參數（約 3.8 億）在輕量級硬體上提供優異的性能。
    
    * Choosing a single token from the probability distribution
        和chapter02 使用model.generate 相比，在03的這一個code block 使用model.model，主要用於提取特徵（feed forward computing）。
        並且和直接 output = model(**tokens)[0] 也不同。這個方法是直接抓最後一層。每一個token 在模型的的最後一層，如果是02的練習，輸入‘hello world' 產生四個token，就會是tensor([[1,4,384(看用什麼模型)]])

        * 解釋model.generate(input_id, max_new_token=20)
        這個方法是一個完整的、迭代的、自回歸過程（Autoregressive）過程。
        模型接收input_id，然後會去預測第一個token，然後會把這一個token加入輸入序列，再去預測下一個token，依此類推，直到預測滿輸入序列長度。
        輸出的內容，可以再透過tokenizer.decode 去做解碼。
        > max_new_tokens: 限制生成的長度。
        > do_sample: 是否使用採樣（如Top-K, Temperature）讓結果更多樣化。
        > num_beams: 控制是否使用Beams search 進行更優質的生成。
        :lamp 為什麼使用beam search 可以產生更優質的生成？
        讓CLM 模型說話的方法。
        目的：執行完整的文本生成過程。
        功能：內部會迭代地執行1→LM Head→選擇下一個詞元→重新輸入，直到達到目標長度。
        輸出：新的詞元ＩＤ序列（包含原始輸入和新生成的文本）
        用途：實際的文本生成、對話、寫作。這是唯一可以讓你直接可讀文本的方法。

        * model.model(input_id)
        目的：特徵提取/前項傳播（feature extraction/ Forward Pass）
        這個方法執行的是，模型單一步驟的前項傳播計算。
        只進行一次計算，讓模型接收input_id。會計算每一個輸入的token在模型的最後一蹭的隱藏狀態。並且不會進任何迭代或是產生新的token。
        輸出的內容，會是一個tuple 和 ModelOutput 物件。
        主要的輸出是 最後隱藏狀態（Last Hidden State）：一個
        這一個輸出是數字向量，不是人類可以讀懂的文本。

            * 總結：
                1. 目的：計算模型的內部特徵（interal feature）
                2. 功能：執行單次且非迭代的前項傳播
                3. 輸出：隱藏狀態(hidden state) ：高維度、抽象的數字向量
                4. 用途：模型微調（finetuning）、特徵提取或是提供下一步LM head的轉換。

        * model.lm_head(model.model(input_id))[0]
            1. 目的：計算下一個**詞源** 在詞彙表上的分數(score/ logits)
            2. 功能：執行model.model 後，接著用線性層（LM head）進行維度投影
            3. 輸出：logits：未標準化的32064維度的數向量。分數最高的就是模型最可能預測的詞元。
            note: model.model的輸出，目前在這一個應用是3072維。
            4. 用途：訓練時計算loss，或在推理時手動選擇下一個詞源。
            
    # 使用model.generate 有沒有使用use_cache的差別






