# Transformers for Language Modeling (Generative AI language modeling: embeddings, attention, the Transformer, and how to train it)

## Table of Contents

- **[Learning Path](#learning-path)**
  - [The big picture](#the-big-picture)
- **[Part 1: The Building Blocks of Language Modeling](#part-1-the-building-blocks-of-language-modeling)**
  - [1.1 Word embeddings: turning words into vectors](#11-word-embeddings-turning-words-into-vectors)
  - [1.2 Positional encoding: giving the model a sense of order](#12-positional-encoding-giving-the-model-a-sense-of-order)
  - [1.3 Putting the two together](#13-putting-the-two-together)
- **[Part 2: The Heart of the Transformer: Self-Attention](#part-2-the-heart-of-the-transformer-self-attention)**
  - [2.1 The goal](#21-the-goal)
  - [2.2 Query, Key, and Value (Q, K, V)](#22-query-key-and-value-q-k-v)
  - [2.3 The process, step by step](#23-the-process-step-by-step)
  - [2.4 Scaled dot-product attention](#24-scaled-dot-product-attention)
  - [2.5 A worked example](#25-a-worked-example)
  - [2.6 The lab's translation-dictionary analogy](#26-the-labs-translation-dictionary-analogy)
  - [2.7 Attention in code](#27-attention-in-code)
  - [2.8 Multi-head attention](#28-multi-head-attention)
- **[Part 3: Building the Transformer Architecture](#part-3-building-the-transformer-architecture)**
  - [3.1 Inside one encoder layer](#31-inside-one-encoder-layer)
  - [3.2 Encoder and decoder](#32-encoder-and-decoder)
  - [3.3 The main PyTorch modules](#33-the-main-pytorch-modules)
  - [3.4 An encoder-only classifier (the M3-L2 lab)](#34-an-encoder-only-classifier-the-m3-l2-lab)
- **[Part 4: Training and Optimization for Large Transformers](#part-4-training-and-optimization-for-large-transformers)**
  - [4.1 The techniques at a glance](#41-the-techniques-at-a-glance)
  - [4.2 Gradient accumulation](#42-gradient-accumulation)
  - [4.3 Mixed-precision training (AMP)](#43-mixed-precision-training-amp)
  - [4.4 Distributed training](#44-distributed-training)
  - [4.5 Efficient optimizers: AdamW and LAMB](#45-efficient-optimizers-adamw-and-lamb)
  - [4.6 The training process in the lab (M3-L2)](#46-the-training-process-in-the-lab-m3-l2)
- **[Part 5: Summary and Self-check](#part-5-summary-and-self-check)**
  - [5.1 Key concepts](#51-key-concepts)
  - [5.2 Common mistakes](#52-common-mistakes)
  - [5.3 Self-check](#53-self-check)

---

## Learning Path

| Part | Topic | The big question | Lab |
|---|---|---|---|
| **1** | Building blocks: embeddings and positional encoding | How does the model get words *and* word order as numbers? | Applying Transformers for Classification (M3-L2); Attention Mechanism and Positional Encoding |
| **2** | Self-attention | How does the model understand context? | Attention Mechanism and Positional Encoding |
| **3** | The Transformer architecture | How do the pieces fit together (encoder, decoder, classifier)? | Applying Transformers for Classification (M3-L2) |
| **4** | Training large Transformers | How do we train big models with limited memory and time? | Applying Transformers for Classification (M3-L2) |
| **5** | Summary and self-check | What should we remember? | |

### The big picture

```text
text → tokens → IDs → embeddings (what the word means)
                    + positional encoding (where the word is)
     → [ self-attention → add & norm → feed-forward → add & norm ] × N layers
     → pooling → linear layer → prediction
```

> **About this manual:** it combines the theory with the code of the two lab notebooks named above. We checked the explanations and formulas, and corrected a few statements (the notes are marked with **Note** or **Careful**). The notebooks themselves were not available for this check, so the lab code below is shown as described in the manual.

---

## Part 1: The Building Blocks of Language Modeling

Before a model can read or generate text, it needs two things: a **numerical meaning** for each word (the embedding) and the **position** of each word in the sentence (the positional encoding).

### 1.1 Word embeddings: turning words into vectors

**The problem.** Computers only understand numbers. A simple idea is to give every word a unique integer ID, but an ID says nothing about meaning. The model would not know that "cat" and "dog" are more related to each other than to "car".

**The solution: word embeddings.** An embedding is a **dense vector** (a list of decimal numbers) for each word. Words with similar meanings end up close together in this "meaning space":

| Word | Example vector | Distance |
|---|---|---|
| cat | `[0.21, -0.45, 0.87, ...]` | |
| dog | similar to the vector for "cat" | **Close** to "cat" |
| car | very different from "cat" | **Far** from "cat" |

> **Careful:** in the first version of this manual, "king" in English and "rey" in Spanish were said to land at the same point. That is true only for **multilingual** embeddings that are trained to line up different languages. Embeddings trained on one language do not do this by themselves.

**How it works in practice.** An `nn.Embedding` layer is a **lookup table**: it takes the integer ID of a word and returns the vector stored for that word. These vectors start as **random numbers** and are **learned during training**. They are not hand-made.

In the `M3-L2 Applying Transformers for Classification` lab:

```python
# A vocabulary is built from the text
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])

# An embedding layer is created with a specified dimension
emsize = 64
embedding = nn.Embedding(vocab_size, emsize)

# The input text is tokenized to indices
my_index = text_pipeline(my_tokens)

# The indices are passed through the embedding layer to get the vectors
my_embeddings = embedding(torch.tensor(my_index))
```

### 1.2 Positional encoding: giving the model a sense of order

**The problem.** A Transformer processes **all the words at the same time** (in parallel). That is great for speed, but it means the model has **no built-in sense of word order**. These two sentences use the same words but mean different things:

- "He painted the **car red**."
- "He painted the **red car**."

Without position information, the model would see them as the same.

**The solution.** We add a **positional encoding** vector to each word embedding. The vector for position `pos` is built from **sine and cosine waves** of different frequencies:

For even dimensions (`2i`), use sine:

$$
PE_{(pos,\,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

For odd dimensions (`2i+1`), use cosine:

$$
PE_{(pos,\,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

Here `pos` is the position of the word (0, 1, 2, ...), `i` numbers the pairs of dimensions, and $d_{model}$ is the embedding size.

**Why this formula works:**

| Property | Why it helps |
|---|---|
| **Every position gets a unique pattern** | The model can tell positions apart |
| **Values stay between −1 and 1** | Sine and cosine are periodic, so the numbers stay small, even for very long sequences |
| **Different frequencies for different dimensions** | Some dimensions change quickly (fine, nearby positions) and others change slowly (coarse, far-apart positions) |

**A tiny example** with $d_{model} = 4$ (each row is the encoding of one position):

| Position | dim 0 (sin) | dim 1 (cos) | dim 2 (sin) | dim 3 (cos) |
|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.0100 | 1.0000 |
| 2 | 0.9093 | −0.4161 | 0.0200 | 0.9998 |
| 3 | 0.1411 | −0.9900 | 0.0300 | 0.9996 |

The first two dimensions change fast from one position to the next, and the last two change slowly. Together they make a different "fingerprint" for each position.

**How it works in practice.** The lab defines a `PositionalEncoding` class that builds the whole table once and adds the right rows to the embeddings:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix 'pe'
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                       # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)             # saved with the model, but not trained

    def forward(self, x):                          # x: (batch, sequence length, d_model)
        # Add the positional encoding to the input embeddings
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
```

> **Note:** the lab's code names the second argument `vocab_size`, but it is really the **maximum sequence length** (`max_len`), because it is the number of positions in the table. We use the clearer name here. Also, this code needs an **even** `d_model` (the sine and cosine halves must have the same size), and sequences **longer than `max_len`** cause an error.
>
> **Check:** `div_term` is just a faster way to write $1/10000^{2i/d_{model}}$. We checked numerically that the vectorized code gives exactly the same table as the formula.

The `Attention Mechanism and Positional Encoding` lab adds the positional encodings to the embeddings and plots them. The plot shows that words keep their closeness in meaning but become distinct by position.

### 1.3 Putting the two together

| Piece | It tells the model | Shape |
|---|---|---|
| **Word embedding** | **What** the word means | (sequence length, $d_{model}$) |
| **Positional encoding** | **Where** the word is | (sequence length, $d_{model}$) |
| **Model input** | **What + where** = embedding **+** positional encoding (added, not joined) | (sequence length, $d_{model}$) |

---

## Part 2: The Heart of the Transformer: Self-Attention

Now each word has a meaning and a position. How does the model understand **context**? The answer is **self-attention**.

### 2.1 The goal

Self-attention lets the model **look at all the other words** in the sequence while it processes one word. It helps the model focus on the most relevant words, resolve ambiguity, and capture relationships. For example, in "The animal didn't cross the street because **it** was tired", attention can link "it" to "animal".

### 2.2 Query, Key, and Value (Q, K, V)

For every token, the model makes three vectors. Each one comes from the token's embedding, multiplied by its own **learned weight matrix** ($W_Q$, $W_K$, $W_V$):

| Vector | Role | Think of it as |
|---|---|---|
| **Query (Q)** | The current token, asking "who is relevant to me?" | A search query |
| **Key (K)** | A label on every token, to be matched against the query | The title of a page |
| **Value (V)** | The actual content of the token, which is retrieved when its key matches | The content of the page |

### 2.3 The process, step by step

1. Compare the **query** of token $i$ with the **key** of every token $j$, using a **dot product**. A bigger score means a better match.
2. **Scale** the scores and apply **softmax**. This gives **attention weights**: positive numbers that add up to 1.
3. The weights decide how much attention token $i$ pays to each token. A higher weight means the value $V_j$ matters more.
4. The output for token $i$ is the **weighted sum of all the value vectors**.

### 2.4 Scaled dot-product attention

The complete operation is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

| Part | What it does |
|---|---|
| $QK^T$ | The similarity score between every query and every key |
| $d_k$ | The size of the key vectors. We divide by its **square root**, $\sqrt{d_k}$ |
| softmax | Turns each row of scores into probabilities (the attention weights) |
| $\cdot\, V$ | Multiplies the weights by the values: a weighted sum |

**Why do we divide by $\sqrt{d_k}$?** When the key vectors are long, the dot products get large. Large scores make softmax almost "all or nothing", and then its gradients are tiny, so learning is hard. Dividing by $\sqrt{d_k}$ keeps the scores in a healthy range. For random vectors, we measured how spread out the dot products are:

| Key size $d_k$ | Spread of $q \cdot k$ (standard deviation) | After dividing by $\sqrt{d_k}$ |
|---:|---:|---:|
| 4 | 2.0 | 1.0 |
| 64 | 8.0 | 1.0 |
| 512 | 22.5 | 1.0 |

And what a large spread does to softmax: the scores `[8, 0, -8]` give the weights `[0.9997, 0.0003, 0.0000]` (the model looks at only one token), while the scaled scores `[1, 0, -1]` give `[0.665, 0.245, 0.090]` (a smoother, more useful mix).

### 2.5 A worked example

Three tokens, with key size $d_k = 2$. Let the queries and keys both be the rows of Q, and the values the rows of V:

$$
Q = K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \qquad
V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 2 & 2 \end{bmatrix}
$$

**Step 1: scores** $QK^T / \sqrt{2}$:

| | token 1 | token 2 | token 3 |
|---|---:|---:|---:|
| **token 1** | 0.707 | 0.000 | 0.707 |
| **token 2** | 0.000 | 0.707 | 0.707 |
| **token 3** | 0.707 | 0.707 | 1.414 |

**Step 2: softmax of each row** gives the attention weights (each row adds up to 1):

| | token 1 | token 2 | token 3 |
|---|---:|---:|---:|
| **token 1** | 0.401 | 0.198 | 0.401 |
| **token 2** | 0.198 | 0.401 | 0.401 |
| **token 3** | 0.248 | 0.248 | 0.504 |

**Step 3: output** = weights × V. For token 1: $0.401 \cdot [1,0] + 0.198 \cdot [0,1] + 0.401 \cdot [2,2] = [1.203, 1.000]$.

| | output |
|---|---|
| **token 1** | [1.203, 1.000] |
| **token 2** | [1.000, 1.203] |
| **token 3** | [1.255, 1.255] |

Each token's output is now a **mix of all tokens' values**, with more weight on the tokens that match its query best.

### 2.6 The lab's translation-dictionary analogy

The `Attention Mechanism...` lab starts with a dictionary made of matrices:

- **K (keys):** each row is the one-hot vector of a French word.
- **V (values):** each row is the one-hot vector of the matching English word.
- **Q (query):** the one-hot vector of the French word to translate.

The translation is `Q @ K.T @ V`:

1. `Q @ K.T` gives a vector with a **1 at the position of the matching row** (and 0 elsewhere): the dictionary lookup.
2. Multiplying by `V` **picks the matching row**: the English word.

This is a **hard lookup**: it only works when the query is exactly one of the keys. The lab then adds a **softmax**, which makes the lookup **soft**:

$$
\text{Output} = \text{softmax}(Q \cdot K^T) \cdot V
$$

Now a query that is *similar* (but not identical) to some keys returns a **blend** of their values, weighted by how well they match. In our small check, a query with a large score for the second key gave the weights `[0.007, 0.987, 0.007]`, almost a hard lookup but still smooth. This soft lookup is exactly what attention does (the full formula also divides by $\sqrt{d_k}$).

### 2.7 Attention in code

```python
import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)      # similarity of every query with every key
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))   # True = this token may not be looked at
    weights = F.softmax(scores, dim=-1)                    # each row adds up to 1
    return weights @ V, weights                            # weighted sum of the values
```

### 2.8 Multi-head attention

Instead of one attention calculation, a Transformer runs **several in parallel**. These are the **heads**.

- **Idea:** the embedding size $d_{model}$ is divided among $h$ heads. Each head works on its own smaller part of size $d_k = d_{model} / h$, with its own $W_Q$, $W_K$, $W_V$. This lets different heads learn **different kinds of relationships** (for example, grammar in one head and meaning in another; this is a common observation, not a rule).
- **Process:**
  1. Project the input into $h$ sets of Q, K, V.
  2. Compute scaled dot-product attention in **each head separately**.
  3. **Concatenate** the outputs of all heads.
  4. A final **linear layer** mixes the information from all heads.

> **Careful:** $d_{model}$ must be **divisible** by the number of heads (for example, 64 with 8 heads gives $d_k = 8$). Otherwise PyTorch raises an error.

In the lab, you can see this idea in a `Head` class, and then in practice with `nn.MultiheadAttention` (which is used inside `nn.TransformerEncoderLayer`):

```python
mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
out, attn_weights = mha(x, x, x)          # self-attention: the query, key, and value are all x
```

---

## Part 3: Building the Transformer Architecture

We now have the building blocks (embeddings, positional encodings, self-attention). This part puts them together.

### 3.1 Inside one encoder layer

One Transformer layer repeats the same small recipe. `nn.TransformerEncoderLayer` is exactly this recipe:

| Step | What happens |
|---:|---|
| 1 | **Multi-head self-attention**: every token looks at all the others |
| 2 | **Add & Norm**: add the layer's input back (a **residual connection**), then apply **layer normalization**. This keeps training stable |
| 3 | **Feed-forward network**: two linear layers with an activation in between, applied to each token separately |
| 4 | **Add & Norm** again |

```text
x → [ self-attention ] → add x → layer norm → [ feed-forward ] → add → layer norm → output
```

`nn.TransformerEncoder` stacks several of these layers. The output of each layer is the input of the next one.

### 3.2 Encoder and decoder

The original Transformer was designed for translation and has **two parts**:

| | **Encoder** | **Decoder** |
|---|---|---|
| **Job** | Read and understand the input sequence | Generate an output sequence, one token at a time |
| **Self-attention** | **Unmasked** (bi-directional): each token can look at all the others, before and after it | **Masked**: each token can look **only at earlier tokens** |
| **Why** | To build a rich, contextual representation of the whole input | The model must not see the answer it is trying to predict |
| **Extra attention** | None | **Encoder-decoder attention** (cross-attention): looks at the encoder's output to stay on topic |
| **PyTorch** | `nn.TransformerEncoderLayer`, `nn.TransformerEncoder` | `nn.TransformerDecoderLayer`, `nn.TransformerDecoder` |
| **Output** | A sequence of context-aware embeddings | The next-token predictions |

**Careful:** modern models often use only one of the two parts. The decoder in the table (with cross-attention) is the **original** one. A **GPT-style** model has no encoder, so it has no cross-attention: it is a stack of masked self-attention layers only.

| Family | Example | Attention | Good for |
|---|---|---|---|
| **Encoder-only** | BERT | Unmasked (sees the whole input) | Understanding: classification, tagging |
| **Decoder-only** | GPT | Masked (sees only the past) | Generating text |
| **Encoder-decoder** | The original Transformer, T5 | Encoder unmasked; decoder masked plus cross-attention | Translation, summarization |

#### The causal mask

To stop a token from seeing the future, we set the scores of the "future" positions to $-\infty$ before the softmax, so their weights become exactly 0. With the three tokens from Section 2.5, the weights become:

| | token 1 | token 2 | token 3 |
|---|---:|---:|---:|
| **token 1** | 1.000 | 0 | 0 |
| **token 2** | 0.330 | 0.670 | 0 |
| **token 3** | 0.248 | 0.248 | 0.504 |

Token 1 can only see itself, token 2 sees tokens 1 and 2, and token 3 sees all three.

```python
n = 3
mask = torch.triu(torch.ones(n, n), diagonal=1).bool()     # True above the diagonal = the future
# or: nn.Transformer.generate_square_subsequent_mask(n)
```

### 3.3 The main PyTorch modules

| Module | What it is |
|---|---|
| `nn.Embedding` | Lookup table: token ID → vector |
| `nn.MultiheadAttention` | Multi-head attention |
| `nn.TransformerEncoderLayer` | One encoder layer: self-attention + feed-forward network (with residuals and layer norm) |
| `nn.TransformerEncoder` | A stack of encoder layers |
| `nn.TransformerDecoderLayer` | One decoder layer (masked self-attention, cross-attention, feed-forward) |
| `nn.TransformerDecoder` | A stack of decoder layers |

### 3.4 An encoder-only classifier (the M3-L2 lab)

The `M3-L2` lab builds an **encoder-only** model for **text classification** (the AG_NEWS news topics):

```python
class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, dim_feedforward, num_layers, num_class, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(embedding_dim, num_class)    # final layer for classification

    def forward(self, x):                                        # x: (batch, sequence length) of token IDs
        x = self.emb(x)                  # 1. Embed
        x = self.pos_encoder(x)          # 2. Add the positional encoding
        x = self.transformer_encoder(x)  # 3. Pass through the encoder layers
        x = x.mean(dim=1)                # 4. Pool (average) over the tokens
        x = self.classifier(x)           # 5. Final classification layer
        return x
```

> **Note:** the lab's constructor arguments are not shown in the manual, so we wrote typical ones here. The important detail is `batch_first=True`. By default, `nn.TransformerEncoderLayer` expects the shape (sequence, batch, features). Our `pos_encoder` and `x.mean(dim=1)` assume (batch, sequence, features), so the two must match.

#### The final linear layer

After the Transformer layers we have one vector for every token, but a classification task needs **one prediction for the whole text**:

- **Pooling:** `x.mean(dim=1)` averages the vectors of all tokens into a single vector for the whole text.
- **Linear layer:** `self.classifier` projects that vector from the hidden size (for example 64) to the number of classes (4 for AG_NEWS). The result is the **logits**, which go to `CrossEntropyLoss`.

> **Two improvements to know about.** (1) If the batch contains **padding**, the mean also averages the padding tokens. Passing a padding mask (`src_key_padding_mask`) and averaging only the real tokens is more accurate. (2) Instead of the mean, BERT-style models add a special **[CLS]** token and classify from its output vector.

---

## Part 4: Training and Optimization for Large Transformers

Large Transformers need a lot of **memory** and **computing time**. The course introduces several techniques to cope.

### 4.1 The techniques at a glance

| Technique | The problem | The idea | PyTorch tool |
|---|---|---|---|
| **Gradient accumulation** | The batch we want does not fit in GPU memory | Add up the gradients of several small batches, then update once | Several `loss.backward()` calls before one `optimizer.step()` |
| **Mixed-precision training** | 32-bit numbers (FP32) use a lot of memory and are slow | Use 16-bit numbers (FP16 or BF16) for most calculations | `torch.autocast` and `GradScaler` (AMP) |
| **Distributed training** | The model or the data is too big for one machine | Use several GPUs or machines | `DistributedDataParallel` |
| **AdamW** | Plain Adam applies weight decay in a less effective way | Apply weight decay separately from the gradient update | `torch.optim.AdamW` |
| **LAMB** | Very large batches make training unstable | Adapt the learning rate for each layer | A library (LAMB is not in core PyTorch) |

### 4.2 Gradient accumulation

Large models train best with large batches, but a large batch may not fit in GPU memory. So we compute the gradients for several **small** batches and **add them up** before updating the weights once. A small GPU then behaves like one with a much bigger batch.

```python
accum_steps = 4                                        # effective batch size = 4 × the mini-batch size
optimizer.zero_grad()
for step, (label, text) in enumerate(train_dataloader):
    loss = criterion(model(text), label) / accum_steps # divide, so the summed gradients are an average
    loss.backward()                                    # gradients keep adding up in .grad
    if (step + 1) % accum_steps == 0:
        optimizer.step()                               # one update for 4 mini-batches
        optimizer.zero_grad()
```

### 4.3 Mixed-precision training (AMP)

Standard training uses 32-bit floating point numbers (FP32). **Mixed precision** uses 16-bit numbers (FP16 or BF16) for most of the calculations, which saves memory and is faster.

**Automatic Mixed Precision (AMP)** in PyTorch decides for us: it runs the fast matrix operations (such as those in attention) in lower precision, and keeps precision-sensitive operations (such as the loss) in FP32. The model weights and the weight updates stay in FP32, so accuracy is preserved. With FP16, a **gradient scaler** multiplies the loss to keep tiny gradients from rounding to zero (BF16 usually does not need it).

```python
scaler = torch.amp.GradScaler("cuda")
for label, text in train_dataloader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):   # fast, lower-precision forward pass
        loss = criterion(model(text), label)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

> **Note:** older tutorials use `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler`. Recent PyTorch versions prefer `torch.autocast` and `torch.amp.GradScaler`, as above.

### 4.4 Distributed training

| Type | How it works | When to use it |
|---|---|---|
| **Data parallelism** | Every GPU holds a **copy of the whole model** and gets a **part of the data**. The gradients are averaged across the GPUs, so all copies stay the same | The model fits on one GPU, but we want to train faster |
| **Model parallelism** | The **model itself is split** across GPUs (for example, different layers on different GPUs) | The model is too big for a single GPU |

In PyTorch, data parallelism is usually done with `DistributedDataParallel` (and a `DistributedSampler` to give each GPU its own part of the data):

```python
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model.to(device), device_ids=[local_rank])
```

### 4.5 Efficient optimizers: AdamW and LAMB

- **AdamW:** a variant of Adam that fixes how **weight decay** (a regularization technique that discourages large weights and helps against overfitting) is applied. It is applied directly to the weights, separately from the adaptive gradient update. This usually gives **better generalization**, and it is a very common choice for Transformers.
- **LAMB:** designed for training with **very large batch sizes**. It adapts the learning rate **layer by layer**, which often means less tuning when the batch size grows.

Two more habits are common in Transformer training: a learning-rate **warm-up** (start small and grow the learning rate for the first steps) and **gradient clipping** (`torch.nn.utils.clip_grad_norm_`).

### 4.6 The training process in the lab (M3-L2)

The lab gives a standard training loop for text classification. It does not use the optimization techniques above, but this is where they would be added.

```python
# In the M3-L2 Lab
LR = 0.1
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)       # the lab uses SGD; AdamW is a common alternative
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

for epoch in tqdm(range(1, EPOCHS + 1)):
    model.train()
    for idx, (label, text) in enumerate(train_dataloader):
        optimizer.zero_grad()

        predicted_label = model(text)
        loss = criterion(predicted_label, label)

        loss.backward()      # gradients are computed here

        # For mixed precision or gradient accumulation, the changes happen here

        optimizer.step()     # the model's weights are updated
```

> **Notes on this loop:**
>
> - The comment in the first version said "an optimizer (e.g., AdamW) is defined here", but the code uses **SGD**.
> - The scheduler is created, but the loop shown **never calls `scheduler.step()`**, so the learning rate never changes. In a full loop we call `scheduler.step()` once per epoch, after the batch loop. (`StepLR` expects an integer step size; `1.0` works but `1` is cleaner.)
> - A learning rate of 0.1 with plain SGD is a simple choice for a small classification demo. Larger Transformers are usually trained with AdamW, a smaller learning rate, and warm-up.

---

## Part 5: Summary and Self-check

### 5.1 Key concepts

| Concept | One-sentence summary |
|---|---|
| **Word embedding** | A learned vector that captures the meaning of a word |
| **Positional encoding** | Sine and cosine waves added to the embeddings to give the model word order |
| **Self-attention** | Lets the model weigh the importance of all the other words when processing one word |
| **Query, key, value** | Queries are matched against keys, and the matching values are mixed together |
| **Scaled dot-product attention** | $\text{softmax}(QK^T/\sqrt{d_k})V$: a weighted sum of values based on query-key similarity |
| **Multi-head attention** | Several attention calculations in parallel, to capture different kinds of relationships |
| **Encoder** | Unmasked self-attention: builds a contextual understanding of the text (BERT-like) |
| **Decoder** | Masked self-attention: generates text and only looks at the past (GPT-like) |
| **Optimization techniques** | Gradient accumulation, mixed precision, distributed training, and AdamW or LAMB make big models trainable |

### 5.2 Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Forgetting the positional encoding | The model cannot tell word order | Add it to the embeddings |
| `batch_first` does not match the rest of the code | Attention mixes the wrong dimensions | Set `batch_first=True` and use (batch, sequence, features) everywhere |
| $d_{model}$ not divisible by the number of heads | PyTorch raises an error | Choose compatible sizes (for example 64 and 8) |
| Sequence longer than `max_len` | The positional table is too short | Increase `max_len` or truncate the text |
| Unmasked attention for text generation | The model "cheats" by seeing the next words | Use the causal mask |
| Averaging over padding tokens | The padding dilutes the pooled vector | Use a padding mask |
| Forgetting `model.eval()` at inference | Dropout stays on, so predictions are noisy | Call `model.eval()` and use `torch.no_grad()` |
| Creating a scheduler but never calling `scheduler.step()` | The learning rate never changes | Call it once per epoch |

### 5.3 Self-check

Try each question first, then open the answer.

**1. Why does a Transformer need positional encoding?**

<details>
<summary>Show answer</summary>

It processes all tokens in parallel, so it has no built-in sense of order. The positional encoding adds the position of each word to its embedding.

</details>

**2. What are the two inputs that are added together to form the model input?**

<details>
<summary>Show answer</summary>

The word embedding (what the word means) and the positional encoding (where the word is).

</details>

**3. What roles do the query, key, and value play?**

<details>
<summary>Show answer</summary>

The query asks "what is relevant to me?", the keys are the labels it is matched against, and the values are the content that is retrieved and mixed according to the match.

</details>

**4. Why do we divide the scores by $\sqrt{d_k}$?**

<details>
<summary>Show answer</summary>

Dot products grow with the key size. Large scores make softmax saturate and give tiny gradients. Dividing keeps the scores in a healthy range.

</details>

**5. Each row of the attention weights adds up to what?**

<details>
<summary>Show answer</summary>

1, because each row is the output of a softmax.

</details>

**6. What is the benefit of multi-head attention? What must be true about $d_{model}$ and the number of heads?**

<details>
<summary>Show answer</summary>

Different heads can learn different kinds of relationships in parallel. $d_{model}$ must be divisible by the number of heads.

</details>

**7. What is the difference between masked and unmasked self-attention, and which one does BERT or GPT use?**

<details>
<summary>Show answer</summary>

Unmasked: each token sees all tokens (BERT, an encoder). Masked: each token sees only earlier tokens (GPT, a decoder), so it cannot see the answer it must predict.

</details>

**8. Does a GPT-style model have encoder-decoder attention?**

<details>
<summary>Show answer</summary>

No. Only the original encoder-decoder Transformer has cross-attention. A GPT-style model has masked self-attention only.

</details>

**9. In the classifier, what do `x.mean(dim=1)` and `self.classifier` do?**

<details>
<summary>Show answer</summary>

The mean pools all token vectors into one vector for the whole text. The linear layer turns that vector into one score per class (the logits).

</details>

**10. How does gradient accumulation let a small GPU act like a bigger one?**

<details>
<summary>Show answer</summary>

It runs several small mini-batches, adding up their gradients (several `backward()` calls), and then does one `optimizer.step()`, so the update uses a larger effective batch.

</details>

**11. What does AdamW fix compared with Adam?**

<details>
<summary>Show answer</summary>

It applies weight decay directly to the weights, separately from the gradient update, which improves generalization.

</details>
