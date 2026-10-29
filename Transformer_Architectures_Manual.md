# Transformer Architectures: A Complete Manual
### Decoder-only (GPT-like), encoder-only (BERT-like), and encoder-decoder (translation) models

> **How to use this manual:** read Parts 1 and 2 first, because they explain what all three families share. Then read Parts 3, 4, and 5 in any order. Part 6 helps us choose, and Part 7 is for revision (glossary, cheat sheet, common mistakes, and a self-check).

## Table of Contents

- **[Learning Path](#learning-path)**
- **[Part 1: Introduction and the Three Families](#part-1-introduction-and-the-three-families)**
  - [1.1 What is a Transformer?](#11-what-is-a-transformer)
  - [1.2 Why Transformers?](#12-why-transformers)
  - [1.3 Where Transformers are used](#13-where-transformers-are-used)
  - [1.4 The three families at a glance](#14-the-three-families-at-a-glance)
- **[Part 2: Core Concepts Shared by All Transformers](#part-2-core-concepts-shared-by-all-transformers)**
  - [2.1 Tokenization](#21-tokenization)
  - [2.2 Vocabulary](#22-vocabulary)
  - [2.3 Token embeddings](#23-token-embeddings)
  - [2.4 Positional encoding](#24-positional-encoding)
  - [2.5 Shape conventions](#25-shape-conventions)
  - [2.6 Self-attention](#26-self-attention)
  - [2.7 Multi-head attention](#27-multi-head-attention)
  - [2.8 Masks](#28-masks)
  - [2.9 Loss functions](#29-loss-functions)
- **[Part 3: Decoder-Only Models (GPT-like)](#part-3-decoder-only-models-gpt-like)**
  - [3.1 Core principle](#31-core-principle)
  - [3.2 Architecture](#32-architecture)
  - [3.3 The model code (`CustomGPTModel`)](#33-the-model-code-customgptmodel)
  - [3.4 Masking in decoder-only models](#34-masking-in-decoder-only-models)
  - [3.5 Data preparation](#35-data-preparation)
  - [3.6 Training](#36-training)
  - [3.7 Inference: autoregressive generation](#37-inference-autoregressive-generation)
  - [3.8 Using a pre-trained GPT-2](#38-using-a-pre-trained-gpt-2)
  - [3.9 Summary: decoder-only models](#39-summary-decoder-only-models)
- **[Part 4: Encoder-Only Models (BERT-like)](#part-4-encoder-only-models-bert-like)**
  - [4.1 Core principle](#41-core-principle)
  - [4.2 Architecture](#42-architecture)
  - [4.3 Task 1: masked language modeling (MLM)](#43-task-1-masked-language-modeling-mlm)
  - [4.4 Task 2: next sentence prediction (NSP)](#44-task-2-next-sentence-prediction-nsp)
  - [4.5 The three BERT embeddings](#45-the-three-bert-embeddings)
  - [4.6 The data preparation pipeline](#46-the-data-preparation-pipeline)
  - [4.7 The BERT model](#47-the-bert-model)
  - [4.8 Training: a combined loss](#48-training-a-combined-loss)
  - [4.9 Using a pre-trained BERT for inference](#49-using-a-pre-trained-bert-for-inference)
  - [4.10 Summary: encoder-only models](#410-summary-encoder-only-models)
- **[Part 5: Encoder-Decoder Models (Translation)](#part-5-encoder-decoder-models-translation)**
  - [5.1 Core principle](#51-core-principle)
  - [5.2 Architecture](#52-architecture)
  - [5.3 The three attention mechanisms](#53-the-three-attention-mechanisms)
  - [5.4 The `Seq2SeqTransformer` model](#54-the-seq2seqtransformer-model)
  - [5.5 Masking in encoder-decoder models](#55-masking-in-encoder-decoder-models)
  - [5.6 Teacher forcing](#56-teacher-forcing)
  - [5.7 The training loop](#57-the-training-loop)
  - [5.8 Inference: greedy decoding](#58-inference-greedy-decoding)
  - [5.9 The translation function](#59-the-translation-function)
  - [5.10 Evaluating with the BLEU score](#510-evaluating-with-the-bleu-score)
  - [5.11 A practical application: translating a PDF](#511-a-practical-application-translating-a-pdf)
  - [5.12 Summary: encoder-decoder models](#512-summary-encoder-decoder-models)
- **[Part 6: Choosing the Right Family](#part-6-choosing-the-right-family)**
  - [6.1 Side-by-side comparison](#61-side-by-side-comparison)
  - [6.2 Decision guide](#62-decision-guide)
- **[Part 7: Reference](#part-7-reference)**
  - [7.1 Glossary](#71-glossary)
  - [7.2 Cheat sheet](#72-cheat-sheet)
  - [7.3 Common mistakes](#73-common-mistakes)
  - [7.4 Self-check](#74-self-check)
  - [7.5 Final words](#75-final-words)
- **[Appendix: What Was Corrected](#appendix-what-was-corrected)**

---

## Learning Path

| Part | Topic | The big question |
|---|---|---|
| **1** | Introduction and the three families | What is a Transformer, and how do GPT, BERT, and translation models differ? |
| **2** | Core concepts shared by all Transformers | Tokens, embeddings, positions, attention, masks, loss |
| **3** | Decoder-only models (GPT-like) | How do we generate text one token at a time? |
| **4** | Encoder-only models (BERT-like) | How do we build a deep understanding of text? |
| **5** | Encoder-decoder models (translation) | How do we map one sequence to another? |
| **6** | Choosing the right family | Which architecture should we use, and when? |
| **7** | Reference | Glossary, cheat sheet, common mistakes, self-check |
| **Appendix** | What was corrected | The errors found in the first version |

> **About this manual:** it covers the three Transformer families with the code of the lab notebooks. We checked the text and the code and fixed a number of errors. Each fix is marked with **Careful** or **Note** where it appears, and the full list is in the Appendix. Code comments that start with `# FIX` show what changed in the code.

---

## Part 1: Introduction and the Three Families

### 1.1 What is a Transformer?

A **Transformer** is a neural network architecture introduced in the 2017 paper *"Attention Is All You Need"* by Vaswani et al. It changed Natural Language Processing (NLP) by replacing recurrent layers (RNNs and LSTMs) with **self-attention**.

### 1.2 Why Transformers?

| Traditional models (RNNs, LSTMs) | Transformers |
|---|---|
| Process words **one after another** | Process the **whole sequence at once** |
| Slow for long sequences | Highly **parallelizable** on GPUs |
| Struggle with long-range dependencies | Every token can look at every other token, so **global context** is easy |
| Hard to train on long texts | Scale well to large datasets |

> **Note:** attention has a price. Comparing every token with every other token costs time and memory that grow with the **square** of the sequence length, so very long inputs are expensive.

### 1.3 Where Transformers are used

| Family | Examples |
|---|---|
| **Decoder-only** | GPT, LLaMA, Granite |
| **Encoder-only** | BERT, RoBERTa |
| **Encoder-decoder** | T5, BART, the original Transformer |

> **Careful:** the first version listed Claude and Gemini as decoder-only. Many modern chat models are decoder-only (for example LLaMA and Granite), but **not every company publishes its architecture**, so we only list models whose design is documented.

### 1.4 The three families at a glance

All Transformers are built from the same blocks. They differ in **how** and **when** attention is applied.

| Family | Example models | Main task | Attention direction | Use cases |
|---|---|---|---|---|
| **Decoder-only** | GPT, LLaMA, Granite | Text **generation** | Causal (left to right) | Chatbots, story writing, code generation, autocomplete |
| **Encoder-only** | BERT, RoBERTa | Text **understanding** | Bidirectional (full context) | Search, sentiment analysis, classification, question answering |
| **Encoder-decoder** | T5, BART, original Transformer | **Sequence-to-sequence** mapping | Encoder bidirectional, decoder causal, plus cross-attention | Translation, summarization, dialogue |

**An analogy.** Imagine three workers who read a book:

- **Decoder-only (GPT):** reads page by page and *predicts the next page*. It only uses what it has already read.
- **Encoder-only (BERT):** reads the whole book at once, looking forward and backward, to understand its *overall meaning*.
- **Encoder-decoder:** first reads and understands the whole book (encoder), then *rewrites it in another language* page by page (decoder), checking the original all the time.

---

## Part 2: Core Concepts Shared by All Transformers

### 2.1 Tokenization

**Tokenization** splits raw text into small units called **tokens** (words, subwords, or characters) that the model can process.

```text
Input text:              "Transformers are amazing."
Basic tokenizer output:  ['transformers', 'are', 'amazing', '.']
```

> **Note:** the lab uses a simple word-level tokenizer. Real models such as BERT and GPT use **subword** tokenizers (WordPiece, byte-pair encoding), which split rare words into smaller pieces so that there are no truly "unknown" words.

**Special tokens** have special jobs:

| Token | Purpose |
|---|---|
| `[PAD]` / `<pad>` | Padding, to make sequences the same length |
| `[UNK]` / `<unk>` | Unknown word |
| `[CLS]` | Start of the sequence; its output is used for classification (BERT) |
| `[SEP]` | Separator between two sentences (BERT) |
| `[MASK]` | A hidden token that the model must predict (BERT) |
| `<bos>` | Beginning of sequence |
| `<eos>` | End of sequence |

### 2.2 Vocabulary

The **vocabulary** is the set of all unique tokens the model knows. Each token has a unique integer index.

```text
{'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4, 'the': 5, 'cat': 6, ...}
```

### 2.3 Token embeddings

A **token embedding** turns the integer index of a token into a dense vector (a list of numbers). Similar words get similar vectors. The embedding layer is a lookup table whose values are **learned** during training.

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
```

The multiplication by $\sqrt{d_{model}}$ comes from the original Transformer. It makes the embeddings larger, so that they are not drowned out when the positional encoding (whose values are between −1 and 1) is added.

### 2.4 Positional encoding

A Transformer processes all tokens **in parallel**, so it has no sense of word order. A **positional encoding** adds information about each token's position to its embedding. There are three common kinds:

| Kind | How it works | Used in |
|---|---|---|
| **Sinusoidal (fixed)** | Sine and cosine waves of different frequencies; nothing to learn | The **original Transformer**; the labs' models |
| **Learned (trainable)** | A position vector for each position, learned during training | **BERT** and **GPT** |
| **Rotary (RoPE)** | Rotates the query and key vectors according to position | Many recent LLMs, such as LLaMA |

> **Careful:** the first version said that sinusoidal encoding is "used in BERT". That is **not** correct: BERT uses **learned** position embeddings. The BERT code in this manual uses the sinusoidal class for simplicity.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)           # (maxlen, 1, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):                       # (sequence length, batch, emb_size)
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
```

> **Note:** this `forward` **adds** the position to its input and applies dropout. Remember this when we build the BERT embedding in Section 4.5.

### 2.5 Shape conventions

The code in this manual is **sequence-first**: tensors have the shape **(sequence length, batch, features)**, because PyTorch's `nn.TransformerEncoderLayer` and `nn.Transformer` use `batch_first=False` by default. Other code (including some tutorials) uses **batch-first**.

| Convention | Shape | Set with |
|---|---|---|
| **Sequence-first** (this manual) | (sequence, batch, features) | the default |
| **Batch-first** | (batch, sequence, features) | `batch_first=True` |

Mixing the two silently mixes up the dimensions, so keep one convention everywhere.

### 2.6 Self-attention

**Self-attention** lets each token "look at" every other token and decide how much attention to pay to each one.

| Vector | Question it answers |
|---|---|
| **Query (Q)** | What am I looking for? |
| **Key (K)** | What do I have? |
| **Value (V)** | What information do I carry? |

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**Why it matters:** in *"The animal didn't cross the street because it was too tired"*, attention can link "it" to "animal", not to "street".

### 2.7 Multi-head attention

Instead of one attention calculation, Transformers run **several in parallel** (the **heads**). Each head works on a smaller part of the embedding, so different heads can learn different kinds of relationships (for example, syntax in one head and meaning in another; this is a common observation, not a rule). The outputs of the heads are joined and mixed by a linear layer. The embedding size must be **divisible** by the number of heads.

### 2.8 Masks

A **mask** controls what each token is allowed to "see".

| Mask | Purpose | Kind | Used in |
|---|---|---|---|
| **Causal mask** | Stops a token from attending to **future** tokens | Attention mask | Decoders (GPT, the translation decoder) |
| **Padding mask** | Ignores `<pad>` tokens | Attention mask | All models |
| **MLM masking** | Hides some input tokens that the model must predict | **Data corruption**, not an attention mask | BERT |

> **Careful:** the first version listed the "MLM mask" next to the attention masks. MLM masking is different: it **replaces tokens in the input** with `[MASK]` (Section 4.3). It does not block any attention.

**The causal mask code:**

```python
def generate_square_subsequent_mask(sz, device=DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
```

A causal mask for 4 tokens (each **row** is a token that is looking; `-inf` means "cannot look at this position"):

```text
[[0, -inf, -inf, -inf],
 [0,   0,  -inf, -inf],
 [0,   0,    0,  -inf],
 [0,   0,    0,    0]]
```

> **Note on conventions:** this mask is an **additive float** mask (0 = allowed, −inf = blocked). PyTorch also accepts a **boolean** mask where `True` means *blocked*, for example `torch.triu(torch.ones(n, n), diagonal=1).bool()`. Both do the same job. The built-in `nn.Transformer.generate_square_subsequent_mask(n)` makes the float version.

### 2.9 Loss functions

| Task | Loss | Note |
|---|---|---|
| **Next-token prediction** (language modeling) | Cross-entropy over the vocabulary | GPT, translation decoder |
| **Masked-token prediction** (MLM) | Cross-entropy over the vocabulary | BERT |
| **Next-sentence prediction** (NSP) | Cross-entropy over 2 classes | BERT |

When we compute the loss, we **ignore the padding tokens** with `ignore_index=PAD_IDX`.

---

## Part 3: Decoder-Only Models (GPT-like)

### 3.1 Core principle

Decoder-only models are trained with **causal language modeling (CLM)**: *predict the next token, given all the tokens before it.*

```text
Given:    [The, cat, sat, on, the]
Predict:  [cat, sat, on, the, mat]
```

At every position, the model must predict the next token. It can only attend to **past and present** tokens, never to future ones. This is enforced by the **causal mask**.

### 3.2 Architecture

```text
Input tokens
    ↓
Token embedding + positional encoding
    ↓
[Transformer block] × N      ← with a causal mask
    ↓
Linear layer (lm_head)
    ↓
Logits over the vocabulary
    ↓
Softmax → next-token probabilities
```

**Key details:**

- It uses the same blocks as an encoder (self-attention + feed-forward network), but with a **causal mask**. There is **no cross-attention**, because there is no encoder.
- It does not need `[CLS]` or `[SEP]` tokens.
- The output layer (`lm_head`) projects each hidden vector to the vocabulary size: one score for every possible next token.

### 3.3 The model code (`CustomGPTModel`)

```python
class CustomGPTModel(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, max_seq_len=500, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout=dropout)

        # Transformer encoder layers (used as a decoder, together with a causal mask)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.embed_size = embed_size
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, x, src_mask=None, key_padding_mask=None):
        # Embed and add positional information
        x = self.embed(x) * math.sqrt(self.embed_size)
        x = self.positional_encoding(x)

        # Generate the causal mask if none is given
        if src_mask is None:
            # FIX: build the mask from the sequence length. The first version called
            # create_mask(x) with the embedded tensor, although that function expects token IDs.
            src_mask = generate_square_subsequent_mask(x.size(0), device=x.device)

        # Pass through the transformer with the causal mask
        output = self.transformer_encoder(x, src_mask, key_padding_mask)
        return self.lm_head(output)
```

> **Note:** the argument `max_seq_len` is not used. The maximum length is set by the positional encoding (`maxlen=5000`). In this lab the positional encoding is the sinusoidal one; GPT models themselves use learned positions.

### 3.4 Masking in decoder-only models

**The causal mask is the defining feature.** Without it, the model could "cheat" during training by looking at the answer.

```python
def create_mask(src, device=DEVICE):
    src_seq_len = src.shape[0]
    src_mask = generate_square_subsequent_mask(src_seq_len)     # triangular causal mask
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)         # True where the token is padding
    return src_mask, src_padding_mask
```

Here `src` must contain **token IDs** (not embeddings), so that the padding positions can be found.

### 3.5 Data preparation

The `get_sample` function creates an input and a target from raw text:

```python
def get_sample(block_size, text):
    sample_leg = len(text)
    random_sample_stop = sample_leg - block_size

    if random_sample_stop >= 1:
        random_start = torch.randint(low=0, high=random_sample_stop, size=(1,)).item()
        stop = random_start + block_size
        src_sequence = text[random_start:stop]
        tgt_sequence = text[random_start + 1:stop + 1]      # shifted by 1!
    elif random_sample_stop <= 0:
        random_start = 0
        stop = sample_leg
        src_sequence = text[random_start:stop]
        tgt_sequence = text[random_start + 1:stop]
        tgt_sequence.append('<|endoftext|>')

    return src_sequence, tgt_sequence
```

**Key insight:** `tgt_sequence` is `src_sequence` **shifted by one position**. That shift turns the data into a next-token prediction task.

### 3.6 Training

```python
loss_fn = CrossEntropyLoss(ignore_index=PAD_IDX)

def train(model, train_data):
    model.train()
    for batch, (src, tgt) in enumerate(train_data):
        logits = model(src, src_mask=None)                       # forward pass
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                       tgt.reshape(-1))                          # compute the loss
        optimizer.zero_grad()
        loss.backward()                                          # backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # gradient clipping
        optimizer.step()                                         # update the weights
```

### 3.7 Inference: autoregressive generation

At inference time, the model produces text **one token at a time**:

```python
def generate(model, prompt=None, max_new_tokens=500, block_size=BLOCK_SIZE):
    model.to(DEVICE)
    prompt_encoded = encode_prompt(prompt).to(DEVICE)
    tokens = []

    for _ in range(max_new_tokens):
        logits = model(prompt_encoded, src_mask=None, key_padding_mask=None)
        logits = logits.transpose(0, 1)
        logit_prediction = logits[:, -1]                        # the logits of the last position
        next_token_encoded = torch.argmax(logit_prediction, dim=-1).reshape(-1, 1)

        if next_token_encoded.item() == EOS_IDX:                # stop at the end-of-sequence token
            break

        prompt_encoded = torch.cat((prompt_encoded, next_token_encoded), dim=0)[-block_size:]
        tokens.append(vocab.get_itos()[next_token_encoded.to('cpu').item()])

    return ' '.join(tokens)
```

**Step by step:**

1. Encode the prompt.
2. Get the logits for the **last** position.
3. Pick the most probable next token (**greedy decoding**).
4. Append it to the prompt (keeping only the last `block_size` tokens).
5. Repeat until `<eos>` or the maximum length.

> **Note:** greedy decoding always takes the top token, so the text can become repetitive. Sampling methods (**temperature**, **top-k**, **top-p**) add controlled randomness (Section 3.8).

### 3.8 Using a pre-trained GPT-2

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_ids = tokenizer.encode("The movie was", return_tensors="pt")
generated_ids = model.generate(
    input_ids,
    max_new_tokens=15,              # FIX: max_length=15 would also count the prompt tokens
    do_sample=True,                 # FIX: temperature, top_k and top_p only work when sampling is on
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

| Setting | What it does |
|---|---|
| `temperature` | Lower (below 1) makes the choice safer and more predictable; higher makes it more varied |
| `top_k` | Choose only among the k most likely tokens |
| `top_p` | Choose among the smallest group of tokens whose probabilities add up to p |

> **Careful:** the first version left out `do_sample=True`. Without it, `generate` uses greedy decoding and **ignores** `temperature`, `top_k`, and `top_p` (the library prints a warning).

### 3.9 Summary: decoder-only models

| Aspect | Details |
|---|---|
| **Attention** | Causal (left to right only) |
| **Training objective** | Next-token prediction (causal language modeling) |
| **Key mask** | Causal (triangular) |
| **Special tokens** | `<bos>`, `<eos>`, `<pad>`, `<unk>` (GPT-2 uses `<\|endoftext\|>` for the start and the end) |
| **Positional encoding** | Learned in GPT models; sinusoidal in this lab's custom model |
| **Use cases** | Text generation, chatbots, code generation |

---

## Part 4: Encoder-Only Models (BERT-like)

### 4.1 Core principle

Encoder-only models build a **deep, bidirectional understanding** of text. They look at the whole sequence at once, with **no causal mask**, and create a rich, contextual embedding for every token.

BERT is pre-trained with **two objectives**:

1. **Masked language modeling (MLM):** predict randomly hidden tokens, using context from **both** directions.
2. **Next sentence prediction (NSP):** predict whether sentence B really follows sentence A.

> **Note:** later models dropped NSP. **RoBERTa**, for example, is trained with MLM only, because NSP was found not to help much. We cover NSP because the original BERT and this lab use it.

### 4.2 Architecture

```text
Input tokens + segment IDs
    ↓
Token embedding + positional encoding + segment embedding
    ↓
[Transformer encoder block] × N      ← no causal mask (bidirectional)
    ↓
    ├──→ [CLS] output   → NSP head → is next? (two classes)
    └──→ all token outputs → MLM head → predictions for the masked tokens
```

### 4.3 Task 1: masked language modeling (MLM)

**Goal:** hide some tokens and predict them from the context on both sides.

```text
Original:  "The cat sat on the mat"
Masked:    "The [MASK] sat on the [MASK]"
Target:    "cat", "mat"
```

**The 80 / 10 / 10 rule.** BERT selects about **15%** of the tokens. Of the selected tokens:

| Share | What happens to the token |
|---:|---|
| **80%** | Replaced by `[MASK]` |
| **10%** | Replaced by a **random token** |
| **10%** | Left **unchanged** (but still predicted) |

**Why?** The token `[MASK]` appears in pre-training, but **never** in the real tasks later (classification, question answering). The 10% random and 10% unchanged cases force the model to build a good representation of **every** token, because it cannot know which ones were changed. This reduces the mismatch between pre-training and later use.

```python
def Masking(token):
    mask = bernoulli_true_false(0.2)   # 20% chance of selection (BERT itself uses 15%)
    if not mask:
        return token, '[PAD]'          # not selected: the label is [PAD], which the loss ignores
    mask_label = token                 # the original token is the label
    r = random.random()
    if r < 0.8:
        token_ = '[MASK]'              # 80%: mask it
    elif r < 0.9:
        rand_idx = torch.randint(0, VOCAB_SIZE, (1,))
        token_ = index_to_en(rand_idx.item())   # 10%: a random token
    else:
        token_ = token                 # 10%: keep it as it is
    return token_, mask_label
```

> **Note:** the code selects **20%** of the tokens, while the text (and BERT) uses **15%**. The lab uses a higher rate, which is a common choice for small datasets. Both are fine, but they should not be confused.

### 4.4 Task 2: next sentence prediction (NSP)

**Goal:** given sentences A and B, predict whether B follows A in the original text.

```text
Input:  [CLS] She enjoys reading books [SEP] He likes playing guitar [SEP]
Label:  is_next = 1   (when the two sentences were really consecutive in the corpus)
```

The final output of the `[CLS]` token goes through a linear layer for this two-class decision.

```python
def process_for_nsp(input_sentences, input_masked_labels):
    bert_input, bert_label, is_next = [], [], []
    available_indices = list(range(len(input_sentences)))

    def build_pair(i, j):
        pair_input = [['[CLS]'] + input_sentences[i] + ['[SEP]'],
                      input_sentences[j] + ['[SEP]']]
        # FIX: the first version never filled bert_label. The MLM labels follow the same layout;
        # [CLS] and [SEP] get the ignored label '[PAD]'.
        pair_label = [['[PAD]'] + input_masked_labels[i] + ['[PAD]'],
                      input_masked_labels[j] + ['[PAD]']]
        return pair_input, pair_label

    while len(available_indices) >= 2:
        if random.random() < 0.5:
            # Case 1: consecutive sentences (is_next = 1)
            index = random.choice(available_indices[:-1])
            pair_input, pair_label = build_pair(index, index + 1)
            is_next.append(1)
            available_indices.remove(index)
            if index + 1 in available_indices:
                available_indices.remove(index + 1)
        else:
            # Case 2: two random sentences (is_next = 0)
            i, j = random.sample(available_indices, 2)
            pair_input, pair_label = build_pair(i, j)
            is_next.append(0)
            available_indices.remove(i)
            available_indices.remove(j)
        bert_input.append(pair_input)
        bert_label.append(pair_label)

    return bert_input, bert_label, is_next
```

> **Careful:** in the first version, `bert_label` stayed **empty**, so the MLM labels were lost and the next stage (`zip(...)`) would produce no training samples. The fix above fills it.
>
> **Note:** a "random" pair can, by chance, be two consecutive sentences, which gives a wrong label 0. With large corpora this is rare.

### 4.5 The three BERT embeddings

BERT combines **three** kinds of embeddings:

| Embedding | Purpose | Example |
|---|---|---|
| **Token embedding** | The meaning of the word | "cat" → a vector |
| **Positional embedding** | The position in the sequence | Position 2 → a vector |
| **Segment embedding** | Which sentence the token belongs to | Sentence A → vector A |

They are **added** to form the input of the encoder stack:

```python
class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.segment_embedding = nn.Embedding(3, emb_size)      # 0 = PAD, 1 = sentence A, 2 = sentence B

    def forward(self, bert_inputs, segment_labels):
        token_emb = self.token_embedding(bert_inputs)
        # FIX: PositionalEncoding already returns (token_emb + position) with dropout applied.
        # The first version added token_emb again, so the token embedding was counted twice.
        x = self.positional_encoding(token_emb) + self.segment_embedding(segment_labels)
        return x
```

> **Careful:** the first version computed `token_emb + self.positional_encoding(token_emb) + segment`, which equals `2 × token_emb + position + segment`. Because `PositionalEncoding.forward` already adds its input, the token embedding must be added only once.

### 4.6 The data preparation pipeline

Preparing data for BERT has **four stages**:

```text
Raw text
   ↓
[1] Tokenization → tokens
   ↓
[2] MLM preparation → masked tokens + labels
   ↓
[3] NSP preparation → sentence pairs + is_next labels
   ↓
[4] Final tensors → input IDs, labels, segment IDs
```

**Stage 4 code:**

```python
def prepare_bert_final_inputs(bert_inputs, bert_labels, is_nexts, to_tensor=True):
    def zero_pad_list_pair(pair_, pad='[PAD]'):
        pair = deepcopy(pair_)
        max_len = max(len(pair[0]), len(pair[1]))
        pair[0].extend([pad] * (max_len - len(pair[0])))
        pair[1].extend([pad] * (max_len - len(pair[1])))
        return pair[0], pair[1]

    flatten = lambda l: [item for sublist in l for item in sublist]
    tokens_to_index = lambda tokens: [vocab[token] for token in tokens]

    bert_inputs_final, bert_labels_final, segment_labels_final, is_nexts_final = [], [], [], []

    for bert_input, bert_label, is_next in zip(bert_inputs, bert_labels, is_nexts):
        segment_label = [[1] * len(bert_input[0]), [2] * len(bert_input[1])]
        bert_input_padded = zero_pad_list_pair(bert_input)
        bert_label_padded = zero_pad_list_pair(bert_label)
        segment_label_padded = zero_pad_list_pair(segment_label, pad=0)

        bert_inputs_final.append(torch.tensor(tokens_to_index(flatten(bert_input_padded)), dtype=torch.int64))
        bert_labels_final.append(torch.tensor(tokens_to_index(flatten(bert_label_padded)), dtype=torch.int64))
        segment_labels_final.append(torch.tensor(flatten(segment_label_padded), dtype=torch.int64))
        is_nexts_final.append(is_next)

    return bert_inputs_final, bert_labels_final, segment_labels_final, is_nexts_final
```

### 4.7 The BERT model

```python
class BERT(torch.nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        super().__init__()
        self.bert_embedding = BERTEmbedding(vocab_size, d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.nextsentenceprediction = nn.Linear(d_model, 2)         # NSP head
        self.masked_language = nn.Linear(d_model, vocab_size)        # MLM head

    def forward(self, bert_inputs, segment_labels):
        padding_mask = (bert_inputs == PAD_IDX).transpose(0, 1)
        my_bert_embedding = self.bert_embedding(bert_inputs, segment_labels)
        transformer_encoder_output = self.transformer_encoder(
            my_bert_embedding, src_key_padding_mask=padding_mask
        )

        # NSP: use the output of the [CLS] token (position 0)
        next_sentence_prediction = self.nextsentenceprediction(transformer_encoder_output[0, :])

        # MLM: predict a token for every position
        masked_language = self.masked_language(transformer_encoder_output)

        return next_sentence_prediction, masked_language
```

The defaults (768 dimensions, 12 layers, 12 heads) are the sizes of **BERT-base**.

### 4.8 Training: a combined loss

BERT is trained on **both** tasks at the same time:

```python
loss_fn_mlm = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
loss_fn_nsp = nn.CrossEntropyLoss()

def train_step(model, bert_inputs, bert_labels, segment_labels, is_nexts):
    optimizer.zero_grad()

    nsp_logits, mlm_logits = model(bert_inputs, segment_labels)

    # Loss for next sentence prediction
    nsp_loss = loss_fn_nsp(nsp_logits, is_nexts)

    # Loss for masked language modeling
    mlm_loss = loss_fn_mlm(mlm_logits.view(-1, mlm_logits.size(-1)), bert_labels.view(-1))

    total_loss = nsp_loss + mlm_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item()
```

### 4.9 Using a pre-trained BERT for inference

```python
from transformers import BertForPreTraining, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')
model.eval()

# Example 1: masked language modeling
masked_text = "The capital of France is [MASK]."
input_ids = tokenizer(masked_text, return_tensors="pt")["input_ids"]

with torch.no_grad():
    outputs = model(input_ids=input_ids)
    predictions = outputs.prediction_logits

mask_positions = input_ids[0] == tokenizer.mask_token_id
predicted_index = torch.argmax(predictions[0, mask_positions], dim=-1).item()    # works for one [MASK]
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
print(f"Predicted token: {predicted_token}")                 # for example: ['paris']

# Example 2: next sentence prediction
text_1 = "The cat sat on the mat"
text_2 = "It was a sunny day"
inputs = tokenizer(text_1, text_2, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    nsp_logits = outputs.seq_relationship_logits

# FIX: in the Hugging Face model, class 0 means "B follows A" and class 1 means "B is a random sentence"
if torch.argmax(nsp_logits, dim=-1).item() == 0:
    print("The model thinks sentence B follows sentence A.")
else:
    print("The model thinks sentence B does NOT follow sentence A.")
```

> **Careful:** the first version printed "NOT consecutive" when the prediction was class 0. In Hugging Face's BERT, the label convention is **0 = B is a continuation of A** and **1 = B is a random sentence**. The reading was inverted.
>
> **Watch the convention:** the lab's own training data uses the **opposite** labels (`is_next = 1` for consecutive sentences). That is fine for a model we train ourselves, but we must not mix the two conventions.

### 4.10 Summary: encoder-only models

| Aspect | Details |
|---|---|
| **Attention** | Bidirectional (no causal mask) |
| **Training objectives** | MLM + NSP (a combined loss); later models such as RoBERTa use MLM only |
| **Masks** | The padding mask for attention; MLM masking is applied to the input data |
| **Special tokens** | `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]` |
| **Embeddings** | Token + positional + segment, **added** together |
| **Positional encoding** | Learned in BERT; sinusoidal in this lab's code |
| **Use cases** | Classification, sentiment analysis, question answering, search |

---

## Part 5: Encoder-Decoder Models (Translation)

### 5.1 Core principle

Encoder-decoder models map an **input sequence** to an **output sequence**:

- an **encoder** reads and understands the whole source sentence;
- a **decoder** writes the target sentence word by word, guided by the encoder's output.

The classic use case is **machine translation**, for example German → English.

### 5.2 Architecture

```text
Source sentence (German)
    ↓
Source tokens + positional encoding
    ↓
[Transformer encoder block] × N      ← bidirectional self-attention
    ↓
Encoder output = "memory"
    ↓ (cross-attention)
Target sentence (English; given as the right answer during training = teacher forcing)
    ↓
Target tokens + positional encoding
    ↓
[Transformer decoder block] × N
    ├── masked self-attention (causal)
    └── cross-attention over the encoder memory
    ↓
Linear layer → logits over the target vocabulary
```

### 5.3 The three attention mechanisms

| Attention | Queries (Q) | Keys (K) and values (V) | Purpose |
|---|---|---|---|
| **Encoder self-attention** | Source tokens | Source tokens | Understand the source sentence |
| **Decoder masked self-attention** | Target tokens | Past target tokens only | Generate the target sequence |
| **Cross-attention** | Target tokens | Source tokens (the memory) | Look at the source while generating |

**Cross-attention is the bridge** between input and output: at every decoding step, it calculates how much attention to pay to each position of the source sentence.

### 5.4 The `Seq2SeqTransformer` model

```python
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size,
                 nhead, src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, trg, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )
```

> **Note:** this lab uses PyTorch's built-in `nn.Transformer`, which already contains both the encoder stack and the decoder stack.

| Argument | Meaning |
|---|---|
| `src_mask` | Which source positions the encoder may look at (all of them, so nothing is masked) |
| `tgt_mask` | The causal mask of the decoder |
| `src_padding_mask`, `tgt_padding_mask` | Which source and target tokens are padding |
| `memory_key_padding_mask` | The padding mask of the source, used by cross-attention |

### 5.5 Masking in encoder-decoder models

We need **three kinds** of mask (the function returns four tensors, because the padding mask is needed for both the source and the target):

```python
def generate_square_subsequent_mask(sz, device=DEVICE):
    """Causal mask for the decoder's self-attention."""
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device=DEVICE):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)                      # causal
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)   # nothing masked

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)                          # padding
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)                          # padding

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
```

**Important:** `src_mask` is **all False** (nothing is masked), because the encoder is allowed to see the whole source sentence.

### 5.6 Teacher forcing

During training, the decoder receives the **correct** target words, not its own predictions. This is **teacher forcing**:

```python
# In the training loop:
tgt_input = tgt[:-1, :]     # remove the last token: this is the decoder input
tgt_out = tgt[1:, :]        # remove the first token: these are the correct answers

logits = model(src, tgt_input, src_mask, tgt_mask,
               src_padding_mask, tgt_padding_mask, src_padding_mask)
loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
```

**Why the shift?** Each position must predict the **next** token:

```text
tgt_input:  [<bos>, w1, w2, w3]
tgt_out:    [w1,    w2, w3, <eos>]
```

Because all positions are given at once (and the causal mask hides the future), the whole sentence is trained **in parallel**.

### 5.7 The training loop

```python
def train_epoch(model, optimizer, train_dataloader):
    model.train()
    losses = 0
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)       # FIX: len(list(train_dataloader)) loaded the whole data a second time
```

### 5.8 Inference: greedy decoding

At inference time there is **no teacher forcing**: the decoder generates tokens **one at a time** and feeds its own predictions back as input.

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    # 1. Encode the source sentence once
    memory = model.encode(src, src_mask)

    # 2. Start with <bos>
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    # 3. Autoregressive loop
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool).to(DEVICE)

        out = model.decode(ys, memory, tgt_mask)               # decode the current sequence
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])                     # logits for the next token
        _, next_word = torch.max(prob, dim=1)                  # greedy choice
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

        if next_word == EOS_IDX:                               # stop at the end
            break

    return ys
```

**Step by step:**

1. **Encode once:** the source sentence is encoded a single time into `memory`.
2. **Start with `<bos>`.**
3. **Loop:** decode the current sequence using the memory, take the prediction for the next token, append it, and stop at `<eos>`.

### 5.9 The translation function

```python
def translate(model, src_sentence):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()

    return " ".join(
        vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))
    ).replace("<bos>", "").replace("<eos>", "")
```

### 5.10 Evaluating with the BLEU score

The **BLEU score** measures translation quality by comparing the generated text with one or more reference translations:

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu_score(generated_translation, reference_translations):
    references = [reference.lower().split() for reference in reference_translations]
    hypothesis = generated_translation.lower().split()
    smooth = SmoothingFunction().method1                 # avoids a score of 0 for short sentences
    return sentence_bleu(references, hypothesis, smoothing_function=smooth)

generated = translate(transformer, "Ein brauner Hund spielt im Schnee .")
references = [
    "A brown dog is playing in the snow .",
    "A brown dog plays in the snow .",
    "A brown dog is frolicking in the snow .",
    "In the snow, a brown dog is playing ."
]
print("BLEU Score:", calculate_bleu_score(generated, references))
```

> **Note:** BLEU compares exact tokens, so the text must have the same case and tokenization. We lowercase both sides and add **smoothing**; without it, a short sentence with no matching four-word group scores almost 0. Note also that `"In the snow, a brown dog..."` has a comma attached to the word "snow,", so it never matches the token "snow" exactly. Tokenize the punctuation consistently.

### 5.11 A practical application: translating a PDF

The lab includes a function that translates a whole PDF from German to English:

```python
def translate_pdf(input_file, translator_model, output_file):
    with pdfplumber.open(input_file) as pdf:
        for page in pdf.pages:
            text_content = page.extract_text()
            sentences = text_content.split(".")
            # ... set up FPDF ...
            for sentence in sentences:
                translated_sentence = translate(translator_model, sentence)
                lines = textwrap.wrap(translated_sentence, width_text)
                for wrap in lines:
                    pdf.cell(0, fontsize_mm, wrap, ln=1)
            pdf.output(output_file, 'F')
```

> **Note:** splitting on "." breaks sentences at abbreviations and decimal numbers, and a model trained on short image captions (as in this kind of lab) usually translates general documents poorly. Treat this as a demo of the pipeline, not as a reliable translator.

### 5.12 Summary: encoder-decoder models

| Aspect | Details |
|---|---|
| **Attention** | Encoder self-attention + decoder masked self-attention + cross-attention |
| **Training** | Teacher forcing (the correct target tokens are the decoder input) |
| **Inference** | Autoregressive greedy decoding |
| **Masks** | Causal (target), padding (source and target) |
| **Special tokens** | `<bos>`, `<eos>`, `<pad>`, `<unk>` |
| **Use cases** | Translation, summarization, dialogue systems, code-to-text |

---

## Part 6: Choosing the Right Family

### 6.1 Side-by-side comparison

| Feature | Decoder-only (GPT) | Encoder-only (BERT) | Encoder-decoder |
|---|---|---|---|
| **Attention type** | Causal (one direction) | Bidirectional | Both |
| **Training objective** | Next-token prediction | MLM (+ NSP in the original) | Sequence-to-sequence prediction |
| **Best for** | Generation | Understanding | Mapping one sequence to another |
| **Special tokens** | `<bos>`, `<eos>` | `[CLS]`, `[SEP]`, `[MASK]` | `<bos>`, `<eos>` |
| **Segment embeddings** | No | Yes (sentence A and B) | No |
| **Causal mask** | Always | Never | Only in the decoder |
| **Cross-attention** | No | No | Yes |
| **Parallel training** | Yes | Yes | Yes (with teacher forcing) |
| **Generating text** | One token at a time | Not designed for it | One token at a time (decoder) |

### 6.2 Decision guide

```text
Do we need to GENERATE text?
│
├── YES → Do we map one sequence to another (a clear input and a clear output)?
│         │
│         ├── YES → ENCODER-DECODER (T5, BART)        e.g. translation, summarization
│         │
│         └── NO  → DECODER-ONLY (GPT, LLaMA)         e.g. chatbot, story writing, autocomplete
│
└── NO  → Do we need to UNDERSTAND or CLASSIFY text?
          │
          └── YES → ENCODER-ONLY (BERT)                e.g. sentiment analysis, search, tagging
```

> **Note:** this guide is a starting point, not a strict rule. Large decoder-only models can also translate, summarize, and classify when we describe the task in the prompt. Encoder-only and encoder-decoder models are still popular when we want a **smaller, cheaper, specialized** model.

---

## Part 7: Reference

### 7.1 Glossary

| Term | Definition |
|---|---|
| **Attention** | A mechanism that lets each token weight the importance of other tokens |
| **Autoregressive** | Generating the output one token at a time, using the previous outputs as input |
| **BLEU score** | A metric for translation quality, comparing to reference translations |
| **Causal language modeling** | Predicting the next token from the previous tokens only |
| **Causal mask** | Prevents a token from attending to future tokens |
| **Cross-attention** | Attention from the decoder to the encoder output (in sequence-to-sequence models) |
| **Cross-entropy loss** | A loss that compares the predicted probabilities with the true labels |
| **Decoder** | The component that generates the output sequence |
| **Encoder** | The component that turns the input sequence into a representation |
| **Fine-tuning** | Taking a pre-trained model and training it further on a specific task |
| **Greedy decoding** | Always choosing the most probable next token |
| **MLM (masked language modeling)** | Pre-training task: predict randomly hidden tokens |
| **NSP (next sentence prediction)** | Pre-training task: predict whether sentence B follows sentence A |
| **Padding** | Adding `<pad>` tokens to make sequences the same length |
| **Positional encoding** | Information about token order, added to the embeddings (sinusoidal, learned, or rotary) |
| **Pre-training** | Training on large unlabeled data to learn general language patterns |
| **Q, K, V** | Query, key, value: the vectors used to compute attention |
| **Segment embedding** | A vector that says which sentence (A or B) a token belongs to |
| **Self-attention** | Attention within one sequence: tokens attend to each other |
| **Teacher forcing** | Giving the decoder the correct tokens during training |
| **Token** | The basic unit of text (word, subword, or character) |
| **Tokenization** | Splitting text into tokens |
| **Transformer** | A neural network architecture based on attention |
| **Vocabulary** | The set of known tokens, mapped to integer indices |

### 7.2 Cheat sheet

#### The three families

```text
DECODER-ONLY (GPT)            ENCODER-ONLY (BERT)             ENCODER-DECODER (translation)
embedding + position          embedding + position + segment  embedding + position
        ↓                              ↓                              ↓
[ blocks × N ]                [ blocks × N ]                  [ encoder × N ] → memory
 with causal mask              bidirectional                          ↓
        ↓                              ↓                       [ decoder × N ]
 linear + softmax              ┌───────┴───────┐                masked self-attention
        ↓                      NSP head     MLM head            + cross-attention
 next token                                                           ↓
                                                               linear + softmax → target tokens
```

#### Key code snippets

**Causal mask (boolean version, `True` = blocked):**

```python
mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
```

**Cross-entropy loss with the padding ignored:**

```python
loss_fn = CrossEntropyLoss(ignore_index=PAD_IDX)
loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
```

**Autoregressive generation (one sequence, greedy):**

```python
for _ in range(max_new_tokens):
    logits = model(sequence)                                  # (sequence length, vocabulary)
    next_token = logits[-1].argmax(dim=-1, keepdim=True)      # FIX: keep a 1-D shape so we can concatenate
    sequence = torch.cat([sequence, next_token])
    if next_token.item() == EOS_IDX:
        break
```

> **Careful:** the first version used `torch.argmax(logits[-1])` and `torch.cat([sequence, next_token])`. The argmax gives a 0-dimensional tensor, and PyTorch cannot concatenate it ("zero-dimensional tensor cannot be concatenated").

#### Special tokens

| Token | GPT | BERT | Seq2seq | Meaning |
|---|:---:|:---:|:---:|---|
| `<pad>` / `[PAD]` | Yes | Yes | Yes | Padding |
| `<unk>` / `[UNK]` | Yes | Yes | Yes | Unknown word |
| `<bos>` | Yes | No | Yes | Beginning of sequence |
| `<eos>` | Yes | No | Yes | End of sequence |
| `[CLS]` | No | Yes | No | Classification token |
| `[SEP]` | No | Yes | No | Sentence separator |
| `[MASK]` | No | Yes | No | Masked token |

GPT-2 itself uses a single token, `<|endoftext|>`, for both the beginning and the end.

#### Training objectives

| Model | Objective | Description |
|---|---|---|
| **GPT** | Causal language modeling | Predict the next token from the previous tokens |
| **BERT** | MLM + NSP | Predict the masked tokens, and whether B follows A |
| **Seq2seq** | Conditional language modeling | Predict the next target token from the source and the previous targets |

#### Attention direction

Each **row** is the token that is looking, and each **column** is the token it may look at.

**Decoder-only (causal):**

| Token ↓ sees → | 1 | 2 | 3 | 4 |
|---|:---:|:---:|:---:|:---:|
| **Token 1** | ✓ | ✗ | ✗ | ✗ |
| **Token 2** | ✓ | ✓ | ✗ | ✗ |
| **Token 3** | ✓ | ✓ | ✓ | ✗ |
| **Token 4** | ✓ | ✓ | ✓ | ✓ |

**Encoder-only (bidirectional):**

| Token ↓ sees → | 1 | 2 | 3 | 4 |
|---|:---:|:---:|:---:|:---:|
| **Token 1** | ✓ | ✓ | ✓ | ✓ |
| **Token 2** | ✓ | ✓ | ✓ | ✓ |
| **Token 3** | ✓ | ✓ | ✓ | ✓ |
| **Token 4** | ✓ | ✓ | ✓ | ✓ |

**Encoder-decoder:** the encoder is bidirectional over the source; the decoder is causal over the target and also uses cross-attention to the whole source.

> **Careful:** the first version drew the decoder-only matrix as an **upper** triangle, which reads as "token 1 sees everything and token 4 sees only itself". That is the opposite of causal attention and of the mask in Section 2.8.

### 7.3 Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Forgetting the causal mask in a decoder | The model "cheats" by seeing the next tokens | Always pass the causal mask |
| Adding the token embedding twice (in the BERT embedding) | The embeddings have the wrong scale | `PositionalEncoding` already adds its input (Section 4.5) |
| Confusing the NSP label convention | The answers are inverted | Hugging Face: 0 = continuation, 1 = random (Section 4.9) |
| Using `temperature`, `top_k`, or `top_p` without `do_sample=True` | The settings are silently ignored | Set `do_sample=True` (Section 3.8) |
| Mixing sequence-first and batch-first tensors | The dimensions are mixed up silently | Keep one convention (Section 2.5) |
| Not ignoring padding in the loss | The model is trained to predict `<pad>` | `ignore_index=PAD_IDX` |
| Applying teacher forcing at inference | There is no correct target at translation time | Decode token by token (Section 5.8) |
| Computing BLEU without lowercasing or smoothing | Scores near 0 for short sentences | Make the text consistent and use smoothing (Section 5.10) |
| Forgetting `model.eval()` at inference | Dropout stays on | Call `model.eval()` and use `torch.no_grad()` |

### 7.4 Self-check

Try each question first, then open the answer.

**1. How are the three Transformer families different in the way they use attention?**

<details>
<summary>Show answer</summary>

Decoder-only: causal attention (each token sees only earlier tokens). Encoder-only: bidirectional attention (each token sees all tokens). Encoder-decoder: a bidirectional encoder, a causal decoder, and cross-attention from the decoder to the encoder.

</details>

**2. Which kind of positional encoding does BERT use?**

<details>
<summary>Show answer</summary>

Learned position embeddings. The original Transformer used sinusoidal encoding; BERT and GPT use learned positions.

</details>

**3. Why is the target in `get_sample` shifted by one position?**

<details>
<summary>Show answer</summary>

At every position, the model must predict the next token. The shift lines up each input token with the token that follows it.

</details>

**4. In MLM, why are some selected tokens replaced by a random token or left unchanged?**

<details>
<summary>Show answer</summary>

`[MASK]` never appears in the real tasks later. Randomizing 10% and leaving 10% unchanged forces the model to build a good representation of every token and reduces the mismatch between pre-training and later use.

</details>

**5. Is MLM masking an attention mask?**

<details>
<summary>Show answer</summary>

No. It changes the input data (tokens are replaced by `[MASK]`). Attention masks (causal and padding) block attention between positions.

</details>

**6. In Hugging Face's BERT next-sentence head, what does class 0 mean?**

<details>
<summary>Show answer</summary>

Sentence B is a continuation of sentence A. Class 1 means B is a random sentence.

</details>

**7. What goes wrong if `BERTEmbedding` computes `token_emb + positional_encoding(token_emb) + segment`?**

<details>
<summary>Show answer</summary>

`positional_encoding` already returns the token embedding plus the position, so the token embedding is counted twice.

</details>

**8. Why does `generate` ignore `temperature` unless `do_sample=True`?**

<details>
<summary>Show answer</summary>

Without sampling, generation is greedy (always the top token), so there is nothing for temperature, top-k, or top-p to change.

</details>

**9. What is cross-attention, and which family has it?**

<details>
<summary>Show answer</summary>

Attention in which the queries come from the decoder's target tokens and the keys and values come from the encoder's output (the memory). Only encoder-decoder models have it.

</details>

**10. How are `tgt_input` and `tgt_out` built for teacher forcing?**

<details>
<summary>Show answer</summary>

`tgt_input = tgt[:-1]` (drop the last token) and `tgt_out = tgt[1:]` (drop the first token), so each position predicts the next token.

</details>

**11. Why do we ignore the padding in the loss?**

<details>
<summary>Show answer</summary>

Padding tokens carry no information. Without `ignore_index=PAD_IDX`, the model would be trained to predict them.

</details>

**12. Can a decoder-only model do translation?**

<details>
<summary>Show answer</summary>

Yes, large decoder-only models can translate when the task is described in the prompt. Encoder-decoder models are still a natural fit, and often smaller and cheaper, for sequence-to-sequence tasks.

</details>

### 7.5 Final words

| Family | Role |
|---|---|
| **Decoder-only (GPT)** | Generation: the foundation of many modern LLMs |
| **Encoder-only (BERT)** | Deep understanding: classification, search, tagging |
| **Encoder-decoder** | Sequence mapping: translation, summarization |

**The principles shared by all three:**

- **Attention** (Q, K, V) connects the tokens.
- **Positional encoding** preserves the word order.
- **Masking** controls what each token can see.
- **Cross-entropy loss** drives the learning.
- **Autoregressive generation** produces the output one token at a time (in the generating models).

---

## Appendix: What Was Corrected

| # | Section | What was wrong in the first version | Fix |
|---:|---|---|---|
| 1 | 1.3 | Claude and Gemini listed as decoder-only | Claude's architecture is not published, so only documented models are listed |
| 2 | 2.4 | Sinusoidal positional encoding said to be used in BERT | BERT uses learned positions; added the three kinds (sinusoidal, learned, rotary) |
| 3 | 2.8 | MLM masking listed as an attention mask | It is data corruption; attention masks are causal and padding |
| 4 | 2.9 | Cross-entropy "for next-token prediction (MLM, LM)" | MLM predicts masked tokens; separated the three tasks |
| 5 | 3.3 | `create_mask(x)` called with embeddings | The causal mask is built from the sequence length |
| 6 | 3.8 | `temperature`, `top_k`, `top_p` without `do_sample=True`; `max_length` counts the prompt | Added `do_sample=True` and `max_new_tokens` |
| 7 | 4.1 | RoBERTa shown with MLM + NSP | Noted that later models dropped NSP |
| 8 | 4.3 | "Prevents the model from becoming lazy" for the 80/10/10 rule | Explained the pre-training / later-use mismatch; noted the code selects 20% while BERT uses 15% |
| 9 | 4.4 | `process_for_nsp` never filled `bert_label` | Filled the labels |
| 10 | 4.5 | `BERTEmbedding` counted the token embedding twice | Removed the extra term |
| 11 | 4.9 | NSP class 0 read as "NOT consecutive" | In Hugging Face, 0 = continuation; also noted the lab's opposite convention |
| 12 | 4.9 | `argmax` without a `dim` for the `[MASK]` prediction | Added `dim=-1` |
| 13 | 5.5 | "Three masks" but four tensors returned | Three kinds of mask, four tensors |
| 14 | 5.7 | `len(list(train_dataloader))` loaded the data a second time | `len(train_dataloader)` |
| 15 | 5.10 | BLEU without smoothing or consistent case and tokenization | Added lowercasing and smoothing, and a note on punctuation |
| 16 | 6.2 | The decision flowchart ignored that decoder-only LLMs also do translation and classification | Added a note |
| 17 | 7.2 | Generation snippet could not concatenate a 0-dimensional tensor | Used `keepdim=True` |
| 18 | 7.2 | Decoder-only attention matrix drawn as an upper triangle | Redrawn as a lower triangle (rows look, columns are looked at) |
| 19 | Table of contents | Inconsistent numbering and anchors | New numbering, Parts 1 to 7 |
