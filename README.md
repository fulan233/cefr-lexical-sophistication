# Fine-grained Sense-aware Lexical Sophistication Indices Based on the CEFR Levels of Word Senses

This project releases the automatic analysis tool proposed in the article:

<em>Anonymous authors. Developing Fine-grained Sense-aware Lexical Sophistication Indices Based on the CEFR Levels of Word Senses. Under Review.</em>

This tool can automatically assign words in a text to different CEFR levels based on their senses used in context, thus more effectively capturing nuanced differences in the degree of sophistication of word senses. On this basis, It further computes 50 fine-grained sense-aware lexical sophistication indices based on the distribution of the CEFR levels of the words in a text.

## Prerequisites
**1.python environment of this study**
*   **`Python 3.6.2`**
*   **`NLTK 3.2.4`**
*   **`numpy 1.19.5`**
*   **`pandas 0.20.3`**
*   **`pytorch 1.3.1`**
*   **`tensorflow 1.10.0`**
*   **`bert-serving 1.10.0`**

**2. Download the pre-trained language model**

In this study, we used the [`uncased BERT-Base`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model to generate deep contextualized word embeddings. More options can be found at https://github.com/google-research/bert.
Since BERT is a deep learning model, it is suggested to use the tool on a **GPU-based** device.

## Automatic analysis 

**Step 1. Start the BERT service.**

```python
bert-serving-start \
    -pooling_strategy NONE \
    -max_seq_len 128 \
    -pooling_layer -1 \
    -device_map 0 \           # please specify the GPU device ID
    -model_dir bert_base \    # please specify the directory of the pre-trained BERT model
    -show_tokens_to_client \
    -priority_batch_size 32   # batch_size is set based on GPU memory, in this study the Nvidia 1080TI (11G memory) is used.
```

**Step 2. Tag the senses for polysemous words.**

```python
python tag_text_server.py
```
In this step, we firstly conduct sentence tokenization for each essay, which can be seen in the folder of **`samples`**. After that, we label the sense for each polysemous word sentence by sentence. The sense information is from [English Vocabulary Profile (EVP) Online](https://englishprofile.org/). The English Vocabulary Profile offers reliable information about which words (and importantly, which meanings of those words) and phrases are known and used by learners at each level of the Common European Framework (CEF).



The sense tagging results can be seen in the folder of **`output`**.

**Step 3. Terminate the BERT service.**

```python
bert-serving-terminate -port 5555
```

**Step 4. Compute the sense-aware lexical sophistication indices.**

```python
python export_csv.py        # the default setting is LazyA1 mode, the default window size is 100.
python export_csv.py mode   # you could specify the mode name: AW/CW/Min/LazyA1.
python export_csv.py mode window_size # you could specify the moving average window size.
```

**Modes**
- AW: computing the indices based on the sense tagging results of all words.
- CW: computing the indices based on the sense tagging results of content words only.
- Min: computing the indices by taking the lowest level of each word.
- LazyA1: If a word's lowest level in EVP is A1, treat it as an A1-level word. For other words, same to AW setting. 

The result can be seen in **`EVP_indices_mode.csv`**.
