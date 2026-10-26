# Transformer Architectures: Decoder-Only, Encoder-Only, and Encoder-Decoder Models

---

## Table of Contents

1. [Introduction to Transformers](#1-introduction-to-transformers)
2. [The Three Transformer Families](#2-the-three-transformer-families)
3. [Core Concepts Across All Transformers](#3-core-concepts-across-all-transformers)
4. [Part I: Decoder-Only Models (GPT-like)](#part-i-decoder-only-models-gpt-like)
5. [Part II: Encoder-Only Models (BERT-like)](#part-ii-encoder-only-models-bert-like)
6. [Part III: Encoder-Decoder Models (Translation)](#part-iii-encoder-decoder-models-translation)
7. [Comparison Table: When to Use Which](#7-comparison-table-when-to-use-which)
8. [Glossary of Key Terms](#8-glossary-of-key-terms)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. Introduction to Transformers

### What is a Transformer?

A **Transformer** is a neural network architecture introduced in the 2017 paper *"Attention is All You Need"* by Vaswani et al. It revolutionized Natural Language Processing (NLP) by replacing recurrent layers (RNNs/LSTMs) with **self-attention mechanisms**.

### Why Transformers?

| Traditional Models (RNNs/LSTMs) | Transformers |
|----------------------------------|--------------|
| Process words **sequentially** (word by word) | Process the **entire sequence at once** |
| Slow for long sequences | Highly **parallelizable** |
| Struggle with long-range dependencies | Capture **global context** easily |
| Hard to train on long texts | Scales well to large datasets |

### The Transformer Revolution

Transformers are the foundation of modern AI models like:
- **GPT** (Generative Pre-trained Transformer) — decoder-only
- **BERT** (Bidirectional Encoder Representations from Transformers) — encoder-only
- **T5, BART** — encoder-decoder
- **LLaMA, Claude, Gemini** — decoder-only

---

## 2. The Three Transformer Families

All transformers are built from the same building blocks, but they differ in **how** and **when** attention is applied.

### 2.1 Overview

| Family | Example Models | Primary Task | Attention Direction | Use Cases |
|--------|---------------|--------------|---------------------|-----------|
| **Decoder-Only** | GPT, LLaMA, Granite | Text **Generation** | Causal (left-to-right) | Chatbots, story writing, code generation, autocomplete |
| **Encoder-Only** | BERT, RoBERTa | Text **Understanding** | Bidirectional (full context) | Search, sentiment analysis, classification, Q&A |
| **Encoder-Decoder** | T5, BART, original Transformer | **Sequence-to-Sequence** mapping | Encoder (bidirectional) + Decoder (causal) | Translation, summarization, dialogue |

### 2.2 Analogy: How Each Model "Thinks"

Imagine three different workers reading a book:

- **Decoder-Only (GPT):** Reads the book one page at a time and *predicts the next page*. It only looks at what it has already read.
- **Encoder-Only (BERT):** Reads the whole book at once, looking at all pages to understand the *overall meaning*. It can look forward and backward.
- **Encoder-Decoder:** First reads and understands the entire book (encoder), then *rewrites it in another language* page by page (decoder), constantly checking the original.

---

## 3. Core Concepts Across All Transformers

Before diving into each family, you need to understand these fundamental concepts.

### 3.1 Tokenization

**Tokenization** is the process of splitting raw text into smaller units called **tokens** (words, subwords, or characters) that the model can process.

**Example:**
```
Input text: "Transformers are amazing."
Basic tokenizer output: ['transformers', 'are', 'amazing', '.']
```

**Special tokens** are added for specific purposes:

| Token | Purpose |
|-------|---------|
| `[PAD]` / `<pad>` | Padding to make sequences the same length |
| `[UNK]` / `<unk>` | Unknown words |
| `[CLS]` | Start of sequence (used in BERT for classification) |
| `[SEP]` | Separator between sentences (used in BERT) |
| `[MASK]` | Masked token (used in BERT MLM) |
| `<bos>` | Beginning of sequence |
| `<eos>` | End of sequence |

### 3.2 Vocabulary

The **vocabulary** is the set of all unique tokens the model knows. Each token is mapped to a unique integer index.

**Example vocabulary:**
```
{'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4, 'the': 5, 'cat': 6, ...}
```

### 3.3 Token Embeddings

**Token embeddings** convert numerical tokens into dense vectors (lists of numbers) in a continuous vector space. Similar words get similar vector representations.

**Key idea:** Each token ID is looked up in an embedding matrix to get its vector representation.

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
```

### 3.4 Positional Encoding

Since transformers process all tokens **in parallel**, they have no inherent sense of word order. **Positional encodings** inject information about the position of each token in the sequence.

**Two types:**

1. **Sinusoidal (fixed):** Used in BERT and the original Transformer
   ```python
   class PositionalEncoding(nn.Module):
       def __init__(self, emb_size, dropout, maxlen=5000):
           super().__init__()
           den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
           pos = torch.arange(0, maxlen).reshape(maxlen, 1)
           pos_embedding = torch.zeros((maxlen, emb_size))
           pos_embedding[:, 0::2] = torch.sin(pos * den)
           pos_embedding[:, 1::2] = torch.cos(pos * den)
           pos_embedding = pos_embedding.unsqueeze(-2)
           self.dropout = nn.Dropout(dropout)
           self.register_buffer('pos_embedding', pos_embedding)

       def forward(self, token_embedding):
           return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
   ```

2. **Trainable (learned):** Used in GPT models — the position vectors are learned during training

### 3.5 Self-Attention

**Self-attention** allows each token to "look at" every other token in the sequence and decide how much attention to pay to each one.

**The Q-K-V mechanism:**
- **Query (Q):** What am I looking for?
- **Key (K):** What do I have?
- **Value (V):** What information do I carry?

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Why it matters:** In the sentence *"The animal didn't cross the street because it was too tired,"* self-attention helps the model understand that "it" refers to "animal," not "street."

### 3.6 Multi-Head Attention

Instead of a single attention mechanism, transformers use **multiple attention heads** in parallel. Each head learns to focus on different aspects of the relationships between tokens.

**Example:** One head might focus on syntax, another on semantic meaning, another on positional proximity.

### 3.7 Masking Strategies

Masking is crucial for controlling what each token can "see."

| Mask Type | Purpose | Used In |
|-----------|---------|---------|
| **Causal Mask** | Prevent attending to future tokens | Decoder (GPT, translation decoder) |
| **Padding Mask** | Ignore `<pad>` tokens | All models |
| **MLM Mask** | Hide tokens for prediction | BERT (encoder) |

**Causal mask code:**
```python
def generate_square_subsequent_mask(sz, device=DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
```

**Visualization of a causal mask (4x4):**
```
[[0, -inf, -inf, -inf],
 [0,   0,  -inf, -inf],
 [0,   0,    0,  -inf],
 [0,   0,    0,    0]]
```
Each row represents a token; `-inf` means "cannot attend to this position."

### 3.8 Loss Functions

- **Cross-Entropy Loss:** Used for both classification (NSP) and next-token prediction (MLM, LM).
- **Ignore Index:** When computing loss, padding tokens are ignored (`ignore_index=PAD_IDX`).

---

## Part I: Decoder-Only Models (GPT-like)

### 4.1 Core Principle

Decoder-only models are trained with **Causal Language Modeling (CLM)**: predict the next token given all previous tokens.

**Training objective:**
```
Given: [The, cat, sat, on, the]
Predict: [cat, sat, on, the, mat]
```

The model is fed each position and must predict the next token. It can only attend to **past and present tokens**, never future ones (enforced by causal masking).

### 4.2 Architecture

```
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
[Transformer Encoder Block] × N   ← Notice: uses causal mask
    ↓
Linear Layer (lm_head)
    ↓
Logits over Vocabulary
    ↓
Softmax → Next-Token Probabilities
```

**Key architectural details:**
- Uses **Transformer Encoder layers** but with a **causal mask** — no cross-attention.
- Does **not** use `[CLS]` or `[SEP]` tokens.
- The output layer (`lm_head`) projects the hidden state to the vocabulary size.

### 4.3 Model Code (`CustomGPTModel`)

```python
class CustomGPTModel(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, max_seq_len=500, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout=dropout)

        # Transformer Encoder layers (used as decoder with causal mask)
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

        # Generate causal mask if not provided
        if src_mask is None:
            src_mask, _ = create_mask(x)

        # Pass through transformer with causal mask
        output = self.transformer_encoder(x, src_mask, key_padding_mask)
        return self.lm_head(output)
```

### 4.4 Masking in Decoder-Only Models

**Causal masking is the defining feature.** Without it, the model could "cheat" during training by looking at the answer.

```python
def create_mask(src, device=DEVICE):
    src_seq_len = src.shape[0]
    src_mask = generate_square_subsequent_mask(src_seq_len)  # Upper triangular
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    return src_mask, src_padding_mask
```

### 4.5 Data Preparation

**The `get_sample` function** creates input/target pairs from raw text:

```python
def get_sample(block_size, text):
    sample_leg = len(text)
    random_sample_stop = sample_leg - block_size

    if random_sample_stop >= 1:
        random_start = torch.randint(low=0, high=random_sample_stop, size=(1,)).item()
        stop = random_start + block_size
        src_sequence = text[random_start:stop]
        tgt_sequence = text[random_start + 1:stop + 1]   # Shifted by 1!
    elif random_sample_stop <= 0:
        random_start = 0
        stop = sample_leg
        src_sequence = text[random_start:stop]
        tgt_sequence = text[random_start + 1:stop]
        tgt_sequence.append('<|endoftext|>')

    return src_sequence, tgt_sequence
```

**Key insight:** `tgt_sequence` is `src_sequence` shifted by one position — this is what makes it a *next-token prediction* task.

### 4.6 Training

```python
loss_fn = CrossEntropyLoss(ignore_index=PAD_IDX)

def train(model, train_data):
    model.train()
    for batch, (src, tgt) in enumerate(train_data):
        logits = model(src, src_mask=None)                # Forward pass
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                       tgt.reshape(-1))                   # Compute loss
        optimizer.zero_grad()
        loss.backward()                                   # Backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
        optimizer.step()                                  # Update weights
```

### 4.7 Inference: Autoregressive Generation

At inference time, the model generates text **one token at a time**:

```python
def generate(model, prompt=None, max_new_tokens=500, block_size=BLOCK_SIZE):
    model.to(DEVICE)
    prompt_encoded = encode_prompt(prompt).to(DEVICE)
    tokens = []

    for _ in range(max_new_tokens):
        logits = model(prompt_encoded, src_mask=None, key_padding_mask=None)
        logits = logits.transpose(0, 1)
        logit_prediction = logits[:, -1]                  # Last token's logits
        next_token_encoded = torch.argmax(logit_prediction, dim=-1).reshape(-1, 1)

        if next_token_encoded.item() == EOS_IDX:          # Stop at end-of-sequence
            break

        prompt_encoded = torch.cat((prompt_encoded, next_token_encoded), dim=0)[-block_size:]
        tokens.append(vocab.get_itos()[next_token_encoded.to('cpu').item()])

    return ' '.join(tokens)
```

**Process step-by-step:**
1. Encode the prompt
2. Get logits for the last position
3. Pick the most probable next token (greedy decoding)
4. Append it to the prompt
5. Repeat

### 4.8 Using a Pre-trained GPT-2

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_ids = tokenizer.encode("The movie was", return_tensors="pt")
generated_ids = model.generate(
    input_ids,
    max_length=15,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

### 4.9 Summary: Decoder-Only Models

| Aspect | Details |
|--------|---------|
| **Attention** | Causal (left-to-right only) |
| **Training Objective** | Next-token prediction (Causal LM) |
| **Key Mask** | Causal (upper-triangular) |
| **Special Tokens** | `<bos>`, `<eos>`, `<pad>`, `<unk>` |
| **Positional Encoding** | Trainable (in GPT) or sinusoidal |
| **Use Cases** | Text generation, chatbots, code generation |

---

## Part II: Encoder-Only Models (BERT-like)

### 5.1 Core Principle

Encoder-only models are trained to build a **deep bidirectional understanding** of text. They look at the entire sequence at once — **no causal masking** — and create rich contextual embeddings for each token.

**Two pre-training objectives:**
1. **Masked Language Modeling (MLM):** Predict randomly masked tokens using context from *both* directions.
2. **Next Sentence Prediction (NSP):** Predict whether sentence B follows sentence A.

### 5.2 Architecture

```
Input Tokens + Segment IDs
    ↓
Token Embedding + Positional Encoding + Segment Embedding
    ↓
[Transformer Encoder Block] × N   ← NO causal mask (bidirectional)
    ↓
    ├──→ [CLS] Token Output → NSP Head → Is Next? (0 or 1)
    └──→ All Token Outputs   → MLM Head → Masked Token Predictions
```

### 5.3 The Two Pre-training Tasks

#### Task 1: Masked Language Modeling (MLM)

**Goal:** Randomly hide some tokens and predict them using context from both sides.

**Example:**
```
Original: "The cat sat on the mat"
Masked:   "The [MASK] sat on the [MASK]"
Model must predict: "cat", "mat"
```

**The 80/10/10 masking rule:**
Of the 15% of tokens selected for masking:
- **80%** → replaced with `[MASK]`
- **10%** → replaced with a **random token**
- **10%** → left **unchanged** (but still predicted)

This prevents the model from becoming lazy when it sees `[MASK]`.

**Code:**
```python
def Masking(token):
    mask = bernoulli_true_false(0.2)   # 20% chance of selection
    if not mask:
        return token, '[PAD]'          # Not selected → no label change
    mask_label = token                 # Original token is the label
    r = random.random()
    if r < 0.8:
        token_ = '[MASK]'              # 80%: mask it
    elif r < 0.9:
        rand_idx = torch.randint(0, VOCAB_SIZE, (1,))
        token_ = index_to_en(rand_idx.item())  # 10%: random token
    else:
        token_ = token                 # 10%: keep as-is
    return token_, mask_label
```

#### Task 2: Next Sentence Prediction (NSP)

**Goal:** Given sentences A and B, predict whether B actually follows A in the original text.

**Example:**
```
Input: [CLS] She enjoys reading books [SEP] He likes playing guitar [SEP]
Label: Is Next = 1 (if these were actually consecutive in the corpus)
```

The `[CLS]` token's final output is used for this classification (via a linear head).

**Code:**
```python
def process_for_nsp(input_sentences, input_masked_labels):
    bert_input, bert_label, is_next = [], [], []
    available_indices = list(range(len(input_sentences)))

    while len(available_indices) >= 2:
        if random.random() < 0.5:
            # Case 1: Consecutive sentences (Is Next = 1)
            index = random.choice(available_indices[:-1])
            bert_input.append([
                ['[CLS]'] + input_sentences[index] + ['[SEP]'],
                input_sentences[index + 1] + ['[SEP]']
            ])
            is_next.append(1)
            available_indices.remove(index)
            if index + 1 in available_indices:
                available_indices.remove(index + 1)
        else:
            # Case 2: Random sentences (Is Next = 0)
            indices = random.sample(available_indices, 2)
            bert_input.append([
                ['[CLS]'] + input_sentences[indices[0]] + ['[SEP]'],
                input_sentences[indices[1]] + ['[SEP]']
            ])
            is_next.append(0)
            available_indices.remove(indices[0])
            available_indices.remove(indices[1])

    return bert_input, bert_label, is_next
```

### 5.4 The Three BERT Embeddings

BERT combines **three** embedding types:

| Embedding | Purpose | Example |
|-----------|---------|---------|
| **Token Embedding** | Semantic meaning of the word | "cat" → vector |
| **Positional Embedding** | Word's position in sequence | Position 2 → vector |
| **Segment Embedding** | Which sentence the token belongs to | Sentence A → vector A |

These are **summed** together to form the final input to the encoder stack:

```python
class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.segment_embedding = nn.Embedding(3, emb_size)   # 0=PAD, 1=Sentence A, 2=Sentence B
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, bert_inputs, segment_labels):
        token_emb = self.token_embedding(bert_inputs)
        x = self.dropout(
            token_emb
            + self.positional_encoding(token_emb)
            + self.segment_embedding(segment_labels)
        )
        return x
```

### 5.5 Data Preparation Pipeline

Preparing data for BERT involves **four stages**:

```
Raw Text
   ↓
[1] Tokenization → tokens
   ↓
[2] MLM preparation → masked tokens + labels
   ↓
[3] NSP preparation → sentence pairs + is_next labels
   ↓
[4] Final tensor prep → input_ids, labels, segment_ids
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

### 5.6 The BERT Model

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

        # NSP: Use the [CLS] token (position 0) output
        next_sentence_prediction = self.nextsentenceprediction(transformer_encoder_output[0, :])

        # MLM: Predict all tokens
        masked_language = self.masked_language(transformer_encoder_output)

        return next_sentence_prediction, masked_language
```

### 5.7 Training: Combined Loss

BERT is trained on **both** tasks simultaneously:

```python
loss_fn_mlm = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
loss_fn_nsp = nn.CrossEntropyLoss()

def train_step(model, bert_inputs, bert_labels, segment_labels, is_nexts):
    optimizer.zero_grad()

    nsp_logits, mlm_logits = model(bert_inputs, segment_labels)

    # Loss for Next Sentence Prediction
    nsp_loss = loss_fn_nsp(nsp_logits, is_nexts)

    # Loss for Masked Language Modeling
    mlm_loss = loss_fn_mlm(mlm_logits.view(-1, mlm_logits.size(-1)), bert_labels.view(-1))

    total_loss = nsp_loss + mlm_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item()
```

### 5.8 Using a Pre-trained BERT for Inference

```python
from transformers import BertForPreTraining, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

# Example 1: Masked Language Modeling
masked_text = "The capital of France is [MASK]."
input_ids = tokenizer(masked_text, return_tensors="pt")["input_ids"]

with torch.no_grad():
    outputs = model(input_ids=input_ids)
    predictions = outputs.prediction_logits

predicted_index = torch.argmax(predictions[0, input_ids[0] == tokenizer.mask_token_id]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
print(f"Predicted token: {predicted_token}")   # Output: ['paris']

# Example 2: Next Sentence Prediction
text_1 = "The cat sat on the mat"
text_2 = "It was a sunny day"
inputs = tokenizer(text_1, text_2, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    nsp_logits = outputs.seq_relationship_logits

if torch.argmax(nsp_logits, dim=-1).item() == 0:
    print("The model thinks these sentences are NOT consecutive.")
else:
    print("The model thinks these sentences are consecutive.")
```

### 5.9 Summary: Encoder-Only Models

| Aspect | Details |
|--------|---------|
| **Attention** | Bidirectional (no causal mask) |
| **Training Objectives** | MLM + NSP (combined loss) |
| **Key Mask** | MLM mask (for hiding tokens); padding mask |
| **Special Tokens** | `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]` |
| **Embeddings** | Token + Positional + Segment (all summed) |
| **Use Cases** | Classification, sentiment analysis, Q&A, search |

---

## Part III: Encoder-Decoder Models (Translation)

### 6.1 Core Principle

Encoder-decoder models map an **input sequence** to an **output sequence**. They combine:
- An **encoder** that reads and understands the entire source sentence
- A **decoder** that generates the target sentence word by word, guided by the encoder's output

**The classic use case: Machine Translation** (e.g., German → English).

### 6.2 Architecture

```
Source Sentence (German)
    ↓
Source Tokens + Positional Encoding
    ↓
[Transformer Encoder Block] × N   ← Bidirectional self-attention
    ↓
Encoder Output = "Memory"
    ↓ (cross-attention)
Target Sentence (English, teacher-forced during training)
    ↓
Target Tokens + Positional Encoding
    ↓
[Transformer Decoder Block] × N
    ├── Masked Self-Attention (causal)
    └── Cross-Attention over Encoder Memory
    ↓
Linear Layer → Logits over Target Vocabulary
```

### 6.3 The Three Attention Mechanisms

The decoder uses **three** distinct attention operations:

| Attention | Queries (Q) | Keys (K) & Values (V) | Purpose |
|-----------|-------------|-----------------------|---------|
| **Encoder Self-Attention** | Source tokens | Source tokens | Understand the source sentence |
| **Decoder Masked Self-Attention** | Target tokens | Past target tokens only | Generate the target sequence |
| **Cross-Attention** | Target tokens | Source tokens (memory) | Attend to the source while generating |

**Cross-attention is the bridge** between input and output. At each decoding step, it calculates how much attention to pay to each position in the source sentence.

### 6.4 The `Seq2SeqTransformer` Model

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

**Note:** This lab uses PyTorch's built-in `nn.Transformer`, which contains both the encoder and decoder stacks.

### 6.5 Masking in Encoder-Decoder Models

**Three masks are needed:**

```python
def generate_square_subsequent_mask(sz, device=DEVICE):
    """Causal mask for decoder self-attention."""
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device=DEVICE):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)           # Causal
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)               # Padding
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)               # Padding

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
```

**Important:** The `src_mask` is **all zeros** (i.e., nothing is masked) because the encoder can see the full source sentence.

### 6.6 Teacher Forcing

During training, the decoder is fed the **actual** target sequence (not its own predictions). This is called **teacher forcing**:

```python
# In the training loop:
tgt_input = tgt[:-1, :]     # Remove last token → input to decoder
tgt_out = tgt[1:, :]        # Remove first token → ground-truth labels

logits = model(src, tgt_input, src_mask, tgt_mask,
               src_padding_mask, tgt_padding_mask, src_padding_mask)
loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
```

**Why shift?** Because of the autoregressive nature:
```
tgt_input:      [<bos>, w1, w2, w3]
tgt_out:        [w1,    w2, w3, <eos>]
```
For each position, the model predicts the *next* token.

### 6.7 Training Loop

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

    return losses / len(list(train_dataloader))
```

### 6.8 Inference: Greedy Decoding

At inference time, there is **no teacher forcing**. The decoder generates tokens **one at a time**, feeding its own previous predictions back as input:

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

        out = model.decode(ys, memory, tgt_mask)              # Decode current sequence
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])                     # Logits for next token
        _, next_word = torch.max(prob, dim=1)                  # Greedy choice
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

        if next_word == EOS_IDX:                               # Stop at end
            break

    return ys
```

**Step-by-step:**
1. **Encode once:** The source sentence is encoded **once** into `memory`.
2. **Start with `<bos>`:** The decoder starts with a beginning-of-sequence token.
3. **Loop:**
   - Decode the current sequence using the memory
   - Get the prediction for the next token
   - Append it to the sequence
   - Stop if it's `<eos>`

### 6.9 Translation Function

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

### 6.10 Evaluating with BLEU Score

The **BLEU score** measures translation quality by comparing generated text to reference translations:

```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu_score(generated_translation, reference_translations):
    references = [reference.split() for reference in reference_translations]
    hypothesis = generated_translation.split()
    return sentence_bleu(references, hypothesis)

generated = translate(transformer, "Ein brauner Hund spielt im Schnee .")
references = [
    "A brown dog is playing in the snow .",
    "A brown dog plays in the snow .",
    "A brown dog is frolicking in the snow .",
    "In the snow, a brown dog is playing ."
]
print("BLEU Score:", calculate_bleu_score(generated, references))
```

### 6.11 Practical Application: Translating a PDF

The lab includes a function to translate an entire PDF document from German to English:

```python
def translate_pdf(input_file, translator_model, output_file):
    with pdfplumber.open(input_file) as pdf:
        for page in pdf.pages:
            text_content = page.extract_text()
            sentences = text_content.split(".")
            # ... setup FPDF ...
            for sentence in sentences:
                translated_sentence = translate(translator_model, sentence)
                lines = textwrap.wrap(translated_sentence, width_text)
                for wrap in lines:
                    pdf.cell(0, fontsize_mm, wrap, ln=1)
            pdf.output(output_file, 'F')
```

### 6.12 Summary: Encoder-Decoder Models

| Aspect | Details |
|--------|---------|
| **Attention** | Self-attention (encoder) + masked self-attention (decoder) + cross-attention |
| **Training** | Teacher forcing (feed ground-truth target tokens) |
| **Inference** | Autoregressive greedy decoding |
| **Masks Needed** | Causal (tgt), padding (src + tgt) |
| **Special Tokens** | `<bos>`, `<eos>`, `<pad>`, `<unk>` |
| **Use Cases** | Translation, summarization, dialogue systems, code-to-text |

---

## 7. Comparison Table: When to Use Which

### 7.1 Side-by-Side Comparison

| Feature | Decoder-Only (GPT) | Encoder-Only (BERT) | Encoder-Decoder |
|---------|-------------------|---------------------|-----------------|
| **Attention Type** | Causal (unidirectional) | Bidirectional | Both |
| **Training Objective** | Next-token prediction | MLM + NSP | Seq2seq prediction |
| **Best For** | Generation | Understanding | Mapping sequences |
| **Special Tokens** | `<bos>`, `<eos>` | `[CLS]`, `[SEP]`, `[MASK]` | `<bos>`, `<eos>` |
| **Segment Embeddings?** | No | Yes (sentence A/B) | No |
| **Causal Mask?** | Always | Never | Only in decoder |
| **Cross-Attention?** | No | No | Yes |
| **Parallel Training?** | Yes | Yes | Yes (with teacher forcing) |

### 7.2 Decision Flowchart

```
Do you need to GENERATE text?
│
├── YES → Do you need to map one sequence to another?
│         │
│         ├── YES → Use ENCODER-DECODER (e.g., T5, BART)
│         │         Examples: translation, summarization
│         │
│         └── NO  → Use DECODER-ONLY (e.g., GPT)
│                   Examples: chatbot, story writing, autocomplete
│
└── NO  → Do you need to UNDERSTAND/CLASSIFY text?
          │
          └── YES → Use ENCODER-ONLY (e.g., BERT)
                    Examples: sentiment analysis, search, Q&A
```

---

## 8. Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Attention** | Mechanism allowing each token to weight the importance of other tokens |
| **Autoregressive** | Generating output one token at a time, using previous outputs as input |
| **BLEU Score** | Metric for translation quality (compares to reference translations) |
| **Causal Mask** | Prevents a token from attending to future tokens |
| **Cross-Attention** | Attention from decoder to encoder output (in seq2seq models) |
| **Cross-Entropy Loss** | Loss function comparing predicted probabilities to true labels |
| **Decoder** | Component that generates the output sequence |
| **Encoder** | Component that processes the input sequence into a representation |
| **Fine-tuning** | Taking a pre-trained model and training it on a specific task |
| **Greedy Decoding** | Always selecting the most probable next token |
| **MLM (Masked Language Modeling)** | Pre-training task: predict randomly masked tokens |
| **NSP (Next Sentence Prediction)** | Pre-training task: predict if sentence B follows A |
| **Padding** | Adding `<pad>` tokens to make sequences equal length |
| **Positional Encoding** | Information about token order added to embeddings |
| **Pre-training** | Training on large unlabeled data to learn general language patterns |
| **Q, K, V** | Query, Key, Value — vectors used in attention computation |
| **Segment Embedding** | Vector indicating which sentence a token belongs to (sentence A or B) |
| **Self-Attention** | Attention within a single sequence (tokens attending to each other) |
| **Teacher Forcing** | Feeding ground-truth tokens to the decoder during training |
| **Token** | Basic unit of text (word, subword, or character) |
| **Tokenization** | Splitting text into tokens |
| **Transformer** | Neural network architecture based on attention mechanisms |
| **Vocabulary** | Set of all known tokens mapped to integer indices |

---

## 9. Quick Reference Cheat Sheet

### 9.1 The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    THREE TRANSFORMER FAMILIES                    │
├─────────────────┬─────────────────────┬─────────────────────────┤
│  DECODER-ONLY   │    ENCODER-ONLY     │     ENCODER-DECODER     │
│    (GPT)        │      (BERT)         │   (Translation)         │
├─────────────────┼─────────────────────┼─────────────────────────┤
│                 │                     │                         │
│  [Embedding]    │  [Embedding +       │  [Embedding + PE]       │
│   + PE          │   PE + Segment]     │       │                 │
│       │         │       │             │       ▼                 │
│       ▼         │       ▼             │  ┌─────────┐            │
│  ┌─────────┐    │  ┌─────────┐        │  │ Encoder │            │
│  │ Encoder │    │  │ Encoder │        │  │ (× N)   │            │
│  │ (× N)   │    │  │ (× N)   │        │  └────┬────┘            │
│  │ + Causal│    │  │ + Bi-   │        │       │ Memory          │
│  │   Mask  │    │  │ direct  │        │       ▼                 │
│  └────┬────┘    │  └────┬────┘        │  ┌─────────┐            │
│       │         │       │             │  │ Decoder │            │
│       ▼         │       ▼             │  │ (× N)   │            │
│  [Linear]       │  ┌────┴────┐        │  │ + Cross │            │
│  [Softmax]      │  │NSP Head │        │  │  Attn   │            │
│       │         │  └─────────┘        │  └────┬────┘            │
│       ▼         │  ┌─────────┐        │       ▼                 │
│  Next Token     │  │MLM Head │        │  [Linear]               │
│                 │  └─────────┘        │  [Softmax]              │
│                 │                     │       ▼                 │
│                 │                     │  Target Sequence        │
└─────────────────┴─────────────────────┴─────────────────────────┘
```

### 9.2 Key Code Snippets

**Causal Mask (for decoder):**
```python
mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
```

**Cross-Entropy Loss with Padding Ignored:**
```python
loss_fn = CrossEntropyLoss(ignore_index=PAD_IDX)
loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
```

**Autoregressive Generation:**
```python
for _ in range(max_new_tokens):
    logits = model(sequence)
    next_token = torch.argmax(logits[-1])
    sequence = torch.cat([sequence, next_token])
    if next_token == EOS_IDX:
        break
```

### 9.3 Special Token Reference

| Token | GPT | BERT | Seq2Seq | Meaning |
|-------|-----|------|---------|---------|
| `<pad>` / `[PAD]` | ✓ | ✓ | ✓ | Padding |
| `<unk>` / `[UNK]` | ✓ | ✓ | ✓ | Unknown word |
| `<bos>` | ✓ | — | ✓ | Beginning of sequence |
| `<eos>` | ✓ | — | ✓ | End of sequence |
| `[CLS]` | — | ✓ | — | Classification token |
| `[SEP]` | — | ✓ | — | Sentence separator |
| `[MASK]` | — | ✓ | — | Masked token |

### 9.4 Training Objectives Summary

| Model | Objective | Description |
|-------|-----------|-------------|
| **GPT** | Causal LM | Predict next token given previous tokens |
| **BERT** | MLM + NSP | Predict masked tokens + predict sentence relationship |
| **Seq2Seq** | Conditional LM | Predict next target token given source + previous targets |

### 9.5 Attention Direction Cheat Sheet

```
Decoder-Only:     Token₁ → Token₂ → Token₃ → Token₄    (left-to-right only)
                  ✓       ✓         ✓         ✓
                  ✗       ✓         ✓         ✓
                  ✗       ✗         ✓         ✓
                  ✗       ✗         ✗         ✓

Encoder-Only:     Token₁ ↔ Token₂ ↔ Token₃ ↔ Token₄    (fully bidirectional)
                  ✓       ✓         ✓         ✓
                  ✓       ✓         ✓         ✓
                  ✓       ✓         ✓         ✓
                  ✓       ✓         ✓         ✓

Encoder-Decoder:  Encoder: Bidirectional over source
                  Decoder: Causal over target + Cross-attention to source
```

---

## Final Words

This manual covers the complete landscape of transformer architectures:

1. **Decoder-Only (GPT):** For creative generation — the foundation of modern LLMs.
2. **Encoder-Only (BERT):** For deep understanding — the workhorse of text classification and search.
3. **Encoder-Decoder (Translation):** For sequence mapping — the bridge between languages and formats.

**The unifying principles across all three:**
- **Attention** (Q, K, V) connects tokens
- **Positional encoding** preserves word order
- **Masking** controls what each token can see
- **Cross-entropy loss** drives learning
- **Autoregressive generation** produces output one token at a time

Master these concepts, and you'll be equipped to work with any modern transformer-based model.
