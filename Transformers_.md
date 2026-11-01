# Transformers

## Table of Contents

- **[How to Read This Manual](#how-to-read-this-manual)**
- **[Learning Path](#learning-path)**
  - [The whole story in one picture](#the-whole-story-in-one-picture)
- **[Part 0: A Beginner's Primer (#part-0-a-beginners-primer)**
  - [0.1 What is a neural network?](#01-what-is-a-neural-network)
  - [0.2 Vectors, matrices, and the dot product](#02-vectors-matrices-and-the-dot-product)
  - [0.3 Probabilities and softmax](#03-probabilities-and-softmax)
  - [0.4 Loss and learning](#04-loss-and-learning)
  - [0.5 Text is a sequence](#05-text-is-a-sequence)
  - [0.6 A tiny PyTorch dictionary](#06-a-tiny-pytorch-dictionary)
- **[Part 1: The Big Picture](#part-1-the-big-picture)**
  - [1.1 What is a Transformer?](#11-what-is-a-transformer)
  - [1.2 The problem Transformers solved](#12-the-problem-transformers-solved)
  - [1.3 Why Transformers?](#13-why-transformers)
  - [1.4 The three families](#14-the-three-families)
  - [1.5 One sentence through a Transformer](#15-one-sentence-through-a-transformer)
- **[Part 2: From Words to Numbers](#part-2-from-words-to-numbers)**
  - [2.1 Tokenization: cutting text into pieces](#21-tokenization-cutting-text-into-pieces)
  - [2.2 Vocabulary: a number for every token](#22-vocabulary-a-number-for-every-token)
  - [2.3 Word embeddings: giving numbers a meaning](#23-word-embeddings-giving-numbers-a-meaning)
  - [2.4 Positional encoding: giving the model a sense of order](#24-positional-encoding-giving-the-model-a-sense-of-order)
  - [2.5 Shapes and conventions](#25-shapes-and-conventions)
- **[Part 3: Attention, the Heart of the Transformer](#part-3-attention-the-heart-of-the-transformer)**
  - [3.1 The idea in plain words](#31-the-idea-in-plain-words)
  - [3.2 A warm-up: a dictionary made of matrices](#32-a-warm-up-a-dictionary-made-of-matrices)
  - [3.3 Query, Key, and Value (Q, K, V)](#33-query-key-and-value-q-k-v)
  - [3.4 The process, step by step](#34-the-process-step-by-step)
  - [3.5 Scaled dot-product attention](#35-scaled-dot-product-attention)
  - [3.6 A worked example](#36-a-worked-example)
  - [3.7 Attention in code](#37-attention-in-code)
  - [3.8 Multi-head attention: several highlighters](#38-multi-head-attention-several-highlighters)
  - [3.9 Masks: who is allowed to look at whom?](#39-masks-who-is-allowed-to-look-at-whom)
  - [3.10 Part 3 summary](#310-part-3-summary)
- **[Part 4: Inside a Transformer](#part-4-inside-a-transformer)**
  - [4.1 Inside one encoder layer](#41-inside-one-encoder-layer)
  - [4.2 Stacking layers](#42-stacking-layers)
  - [4.3 Encoder and decoder](#43-encoder-and-decoder)
  - [4.4 What sits on top of the Transformer](#44-what-sits-on-top-of-the-transformer)
  - [4.5 The main PyTorch modules](#45-the-main-pytorch-modules)
- **[Part 5: Training and Generating](#part-5-training-and-generating)**
  - [5.1 What do we train the model to do?](#51-what-do-we-train-the-model-to-do)
  - [5.2 Loss functions](#52-loss-functions)
  - [5.3 The training loop](#53-the-training-loop)
  - [5.4 Teacher forcing and the one-step shift](#54-teacher-forcing-and-the-one-step-shift)
  - [5.5 Training big models: the optimization toolbox](#55-training-big-models-the-optimization-toolbox)
  - [5.6 Generating text](#56-generating-text)
  - [5.7 How do we know the model is good?](#57-how-do-we-know-the-model-is-good)
  - [5.8 Part 5 summary](#58-part-5-summary)
- **[Part 6: Decoder-Only Models (GPT-like)](#part-6-decoder-only-models-gpt-like)**
  - [6.1 Core principle](#61-core-principle)
  - [6.2 Architecture](#62-architecture)
  - [6.3 The model code (`CustomGPTModel`)](#63-the-model-code-customgptmodel)
  - [6.4 Masking in decoder-only models](#64-masking-in-decoder-only-models)
  - [6.5 Data preparation](#65-data-preparation)
  - [6.6 Training](#66-training)
  - [6.7 Inference: autoregressive generation](#67-inference-autoregressive-generation)
  - [6.8 Using a pre-trained GPT-2](#68-using-a-pre-trained-gpt-2)
  - [6.9 Summary: decoder-only models](#69-summary-decoder-only-models)
- **[Part 7: Encoder-Only Models (BERT-like)](#part-7-encoder-only-models-bert-like)**
  - [7.1 Core principle](#71-core-principle)
  - [7.2 Architecture](#72-architecture)
  - [7.3 Task 1: masked language modeling (MLM)](#73-task-1-masked-language-modeling-mlm)
  - [7.4 Task 2: next sentence prediction (NSP)](#74-task-2-next-sentence-prediction-nsp)
  - [7.5 The three BERT embeddings](#75-the-three-bert-embeddings)
  - [7.6 The data preparation pipeline](#76-the-data-preparation-pipeline)
  - [7.7 The BERT model](#77-the-bert-model)
  - [7.8 Training: a combined loss](#78-training-a-combined-loss)
  - [7.9 Using a pre-trained BERT for inference](#79-using-a-pre-trained-bert-for-inference)
  - [7.10 Summary: encoder-only models](#710-summary-encoder-only-models)
- **[Part 8: Hands-On: A Transformer Text Classifier](#part-8-hands-on-a-transformer-text-classifier)**
  - [8.1 The task](#81-the-task)
  - [8.2 The recipe](#82-the-recipe)
  - [8.3 Batch-first positional encoding](#83-batch-first-positional-encoding)
  - [8.4 The classifier model](#84-the-classifier-model)
  - [8.5 The training process in the lab](#85-the-training-process-in-the-lab)
  - [8.6 Part 8 summary](#86-part-8-summary)
- **[Part 9: Encoder-Decoder Models (Translation)](#part-9-encoder-decoder-models-translation)**
  - [9.1 Core principle](#91-core-principle)
  - [9.2 Architecture](#92-architecture)
  - [9.3 The three attention mechanisms](#93-the-three-attention-mechanisms)
  - [9.4 The `Seq2SeqTransformer` model](#94-the-seq2seqtransformer-model)
  - [9.5 Masking in encoder-decoder models](#95-masking-in-encoder-decoder-models)
  - [9.6 Teacher forcing](#96-teacher-forcing)
  - [9.7 The training loop](#97-the-training-loop)
  - [9.8 Inference: greedy decoding](#98-inference-greedy-decoding)
  - [9.9 The translation function](#99-the-translation-function)
  - [9.10 Evaluating with the BLEU score](#910-evaluating-with-the-bleu-score)
  - [9.11 A practical application: translating a PDF](#911-a-practical-application-translating-a-pdf)
  - [9.12 Summary: encoder-decoder models](#912-summary-encoder-decoder-models)
- **[Part 10: Choosing the Right Family](#part-10-choosing-the-right-family)**
  - [10.1 Side-by-side comparison](#101-side-by-side-comparison)
  - [10.2 Decision guide](#102-decision-guide)
  - [10.3 What all three families share](#103-what-all-three-families-share)
- **[Part 11: Reference](#part-11-reference)**
  - [11.1 The big ideas in twelve lines](#111-the-big-ideas-in-twelve-lines)
  - [11.2 Glossary](#112-glossary)
  - [11.3 Cheat sheet](#113-cheat-sheet)
  - [11.4 Common mistakes](#114-common-mistakes)
  - [11.5 Self-check](#115-self-check)
  - [11.6 Final words](#116-final-words)
- **[Appendix: Sources and Corrections](#appendix-sources-and-corrections)**
  - [A. What was merged](#a-what-was-merged)
  - [B. Corrections kept in this manual](#b-corrections-kept-in-this-manual)
  - [C. Which lab covers what](#c-which-lab-covers-what)

---

## How to go through This Manual

**Who is it for?** Anyone who wants to understand Transformers, even if this is the first time. We explain every idea in simple words, with small examples, pictures made of text, and tables. Basic Python helps for the code, but the explanations stand on their own.

| If we want to... | Read |
|---|---|
| Understand everything from zero | Parts 0 to 5 in order, then Parts 6 to 9 |
| Understand the ideas, with little code | Parts 0 to 4, then Part 10 |
| Build and train models in PyTorch | Parts 5 to 9 |
| Revise quickly | Part 11 (glossary, cheat sheet, mistakes, self-check) |

**How to read the code.**

- Code boxes show real PyTorch code from the labs. Comments explain the shapes and the ideas.
- A comment that starts with **`# FIX`** marks a mistake that existed in an earlier version of the code and is now corrected. These comments teach common pitfalls.
- The code needs PyTorch. Names such as `DEVICE`, `PAD_IDX`, `vocab`, and `optimizer` are defined elsewhere in the labs.


## Learning Path

| Part | Topic | The big question |
|---|---|---|
| **0** | Before we start | What math and ideas do we need (neural networks, vectors, softmax, loss)? |
| **1** | The big picture | What is a Transformer, and why was it invented? |
| **2** | From words to numbers | How does a model read text? (tokens, embeddings, positions) |
| **3** | Attention | How does the model understand context? |
| **4** | Inside a Transformer | How do the pieces fit into layers, encoders, and decoders? |
| **5** | Training and generating | How does a Transformer learn, and how does it write text? |
| **6** | Decoder-only models (GPT-like) | How do we generate text one token at a time? |
| **7** | Encoder-only models (BERT-like) | How do we understand text deeply? |
| **8** | Hands-on: a text classifier | How do we build a classifier with an encoder? |
| **9** | Encoder-decoder models (translation) | How do we turn one sequence into another? |
| **10** | Choosing the right family | Which model do we use for which job? |
| **11** | Reference | Glossary, cheat sheet, common mistakes, self-check |
| **Appendix** | Sources and corrections | What was merged and what was fixed |

### The whole story in one picture

```text
text → tokens → numbers (IDs) → embeddings (meaning) + positions (order)
     → [ attention → add & norm → feed-forward → add & norm ] × N layers
     → output layer → prediction (next word, masked word, class, translation)
```

Every part of this manual explains one piece of this picture.

---

## Part 0: A Beginner's Primer

If some words in this part are new, good. They are the only prerequisites, and we explain them in plain language.

### 0.1 What is a neural network?

A **neural network** is a big math function with lots of **adjustable numbers**, called **weights**. It takes numbers in and gives numbers out.

How does it learn? Like a student with a practice exam:

1. We show it an example and it makes a **guess**.
2. We measure **how wrong** the guess was. This measurement is called the **loss**.
3. We **nudge the weights** a little, in the direction that would have made the guess less wrong.
4. We repeat this millions of times, with many examples.

After enough rounds, the weights hold useful knowledge, and the guesses become good.

### 0.2 Vectors, matrices, and the dot product

| Word | Plain meaning | Example |
|---|---|---|
| **Vector** | A list of numbers | `[0.2, 0.8, -0.5]` |
| **Matrix** | A grid of numbers (a table) | 2 rows × 3 columns |
| **Dot product** | Multiply two vectors position by position, then add everything up | `[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 32` |

The dot product is a **similarity score**: it is large when two vectors point in a similar direction, and small (or negative) when they do not. Attention (Part 3) is built on this idea.

**Multiplying a vector by a matrix** simply means doing many dot products at once: one for each row of the matrix. Neural networks are mostly this operation, repeated.

### 0.3 Probabilities and softmax

A model usually ends with a list of raw scores, called **logits**. Raw scores are hard to read: they can be any size, positive or negative. **Softmax** turns them into **probabilities** that are positive and add up to 1:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

| | Score 1 | Score 2 | Score 3 | Sum |
|---|---:|---:|---:|---:|
| **Raw scores (logits)** | 2.0 | 1.0 | 0.1 | |
| **After softmax** | 0.66 | 0.24 | 0.10 | 1.00 |

The biggest score gets the biggest probability. We will use softmax again and again: to choose the next word, and to decide how much attention to pay to each word.

### 0.4 Loss and learning

For choosing among classes or words, we use the **cross-entropy loss**. It looks at the probability the model gave to the **correct** answer:

| Probability given to the correct answer | Loss | Meaning |
|---:|---:|---|
| 0.9 | 0.105 | Confident and right: small loss |
| 0.5 | 0.693 | Unsure: medium loss |
| 0.1 | 2.303 | Confident and wrong: big loss |

**Gradient descent** is the nudging step. The **gradient** tells us, for every weight, which direction makes the loss smaller. The **learning rate** is the size of each nudge: too big and we overshoot, too small and we learn slowly.

| Training word | Meaning |
|---|---|
| **Epoch** | One full pass through all the training data |
| **Batch** | A small group of examples processed together |
| **Optimizer** | The tool that does the nudging (for example SGD or AdamW) |

### 0.5 Text is a sequence

Text is a **sequence** of pieces (words). The **order matters**: "dog bites man" and "man bites dog" use the same words but mean different things. A good language model must understand both the words and their order.

### 0.6 A tiny PyTorch dictionary

| Term | Meaning |
|---|---|
| **Tensor** | A multi-dimensional array of numbers (a vector, a matrix, or bigger) |
| **Shape** | The size of each dimension, for example `(10, 32, 64)` = 10 steps, 32 sentences, 64 numbers each |
| **`nn.Module`** | A building block of a model (a class with a `forward` method) |
| **`nn.Linear`** | A layer that multiplies by a weight matrix |
| **`nn.Embedding`** | A lookup table from word IDs to vectors |
| **`loss.backward()`** | Computes the gradients |
| **`optimizer.step()`** | Nudges the weights using the gradients |
| **`.to(device)`** | Moves a tensor or model to the CPU or the GPU |

---

## Part 1: The Big Picture

### 1.1 What is a Transformer?

A **Transformer** is a neural network design introduced in 2017 in the paper *"Attention Is All You Need"* by Vaswani and colleagues. It changed Natural Language Processing (NLP), the field of making computers work with human language. Its main idea is **self-attention**: letting every word look at every other word to understand the context.

Transformers are the foundation of GPT, BERT, and most modern language models.

### 1.2 The problem Transformers solved

Before Transformers, language models used **recurrent neural networks (RNNs and LSTMs)**. An RNN reads a sentence **one word at a time**, keeping a small memory of what it has read. This causes problems:

- **It is slow.** Word 5 must wait for word 4, which waits for word 3, and so on. We cannot use the GPU to process the words at the same time.
- **The memory fades.** By the end of a long sentence, the model may have forgotten the beginning.
- **It is hard to train on long texts.**

> **An analogy.** An RNN reads a long letter through a small keyhole, one word at a time, and tries to remember everything. A Transformer lays the whole letter on the table, so it can look at any part, at any time.

### 1.3 Why Transformers?

| Traditional models (RNNs, LSTMs) | Transformers |
|---|---|
| Process words **one after another** | Process the **whole sequence at once** |
| Slow for long sequences | Highly **parallelizable** on GPUs |
| Struggle with long-range dependencies | Every word can look at every other word, so **global context** is easy |
| Hard to train on long texts | Scale well to large datasets |

> **Note:** attention has a price. Comparing every token with every other token costs time and memory that grow with the **square** of the text length. Very long inputs are expensive.

### 1.4 The three families

All Transformers use the same building blocks. They differ in **how** and **when** attention is used. The result is three families:

| Family | Example models | Main job | Attention direction | Use cases |
|---|---|---|---|---|
| **Decoder-only** | GPT, LLaMA, Granite | **Generating** text | Causal (left to right) | Chatbots, story writing, code generation, autocomplete |
| **Encoder-only** | BERT, RoBERTa | **Understanding** text | Bidirectional (sees everything) | Search, sentiment analysis, classification, question answering |
| **Encoder-decoder** | T5, BART, the original Transformer | **Turning one sequence into another** | Encoder bidirectional, decoder causal, plus cross-attention | Translation, summarization, dialogue |

**An analogy.** Imagine three workers who read a book:

- **Decoder-only (GPT):** reads page by page and *predicts the next page*. It only uses what it has already read.
- **Encoder-only (BERT):** reads the whole book at once, looking forward and backward, to understand its *overall meaning*.
- **Encoder-decoder:** first reads and understands the whole book (the encoder), then *rewrites it in another language*, page by page (the decoder), checking the original all the time.

> **Careful:** earlier versions listed Claude and Gemini as decoder-only. Many modern chat models are decoder-only (for example LLaMA and Granite), but **not every company publishes its architecture**, so we only name models whose design is documented.

### 1.5 One sentence through a Transformer

Let us follow the sentence **"The cat sat"** through a Transformer. Each step has its own part in this manual.

| Step | What happens | Part |
|---:|---|---|
| 1 | The text is cut into **tokens**: `the`, `cat`, `sat` | 2.1 |
| 2 | Each token becomes an **ID number** (its place in the vocabulary) | 2.2 |
| 3 | Each ID becomes an **embedding**: a vector that stands for the word's meaning | 2.3 |
| 4 | A **positional encoding** is added, so the model knows who is first, second, third | 2.4 |
| 5 | **Attention** lets each word look at the other words and mix in useful information | 3 |
| 6 | A small **feed-forward network** processes each word further. Steps 5 and 6 repeat in several **layers** | 4 |
| 7 | The final **output layer** gives a score for every possible answer (for example every word that could come next) | 5, 6 |
| 8 | **Softmax** turns the scores into probabilities, and we pick the answer | 0.3 |

---

## Part 2: From Words to Numbers

A computer cannot read words. It can only do math on numbers. So the first job of every language model is to turn text into numbers in a smart way. There are four steps: **tokens → IDs → embeddings → positions**.

### 2.1 Tokenization: cutting text into pieces

**Tokenization** splits raw text into small units called **tokens**. A token can be a word, a part of a word (a **subword**), or a single character.

```text
Input text:              "Transformers are amazing."
Basic tokenizer output:  ['transformers', 'are', 'amazing', '.']
```

> **Note:** the labs use a simple word-level tokenizer. Real models such as BERT and GPT use **subword** tokenizers (WordPiece, byte-pair encoding). They cut a rare word into smaller known pieces, for example "unbelievable" into "un" + "believ" + "able". This way the model almost never meets a truly unknown word.

**Special tokens** are extra tokens with special jobs:

| Token | Purpose |
|---|---|
| `[PAD]` / `<pad>` | **Padding**, to make all sentences in a batch the same length |
| `[UNK]` / `<unk>` | An **unknown** word that is not in the vocabulary |
| `[CLS]` | The "classification" token at the start; its output summarizes the whole input (BERT) |
| `[SEP]` | A **separator** between two sentences (BERT) |
| `[MASK]` | A hidden word that the model must guess (BERT) |
| `<bos>` | **B**eginning **o**f **s**equence |
| `<eos>` | **E**nd **o**f **s**equence |

### 2.2 Vocabulary: a number for every token

The **vocabulary** is the list of all the unique tokens the model knows. Each token gets a unique whole number, its **ID**.

```text
{'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4, 'the': 5, 'cat': 6, ...}
```

Now the sentence "the cat" becomes the numbers `[5, 6]`. A word that is not in the vocabulary becomes the `[UNK]` token.

### 2.3 Word embeddings: giving numbers a meaning

**The problem.** An ID such as 6 for "cat" says nothing about the word. ID 6 is not "closer" to ID 5 ("the") than to ID 4000. The model would not know that "cat" and "dog" are related.

**The solution: word embeddings.** An **embedding** is a **vector** (a list of numbers) for each word. Think of it as the word's **coordinates on a map of meaning**. Words with similar meanings are placed close together:

| Word | Example vector | On the map |
|---|---|---|
| cat | `[0.21, -0.45, 0.87, ...]` | |
| dog | similar to the vector for "cat" | **Close** to "cat" |
| car | very different from "cat" | **Far** from "cat" |

The vector has many numbers (for example 64, 512, or 768). This length is called the **embedding size** (or $d_{model}$).

> **Careful:** an earlier version said that "king" (English) and "rey" (Spanish) land at the same point. That is true only for **multilingual** embeddings that are trained to line up different languages. Embeddings trained on one language do not do this by themselves.

**How it works.** An `nn.Embedding` layer is a **lookup table**: one row for each word in the vocabulary. We give it a word ID, and it returns the row (the vector) for that word. The rows start as **random numbers** and are **learned during training**. Nobody writes them by hand.

```text
Embedding table (vocabulary size × embedding size)

ID 0  [PAD]   → [ 0.01, -0.30,  0.12, ... ]
ID 5  "the"   → [ 0.44,  0.10, -0.02, ... ]
ID 6  "cat"    → [ 0.21, -0.45,  0.87, ... ]     ← the model looks up row 6
```

In the lab for the text classifier (M3-L2):

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

The embedding layer used in the GPT, BERT, and translation labs multiplies the result by $\sqrt{d_{model}}$:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
```

This scaling comes from the original Transformer. It makes the embeddings larger, so that they are not drowned out when the positional encoding (values between −1 and 1) is added on top.

### 2.4 Positional encoding: giving the model a sense of order

**The problem.** A Transformer processes **all words at the same time**, in parallel. That is great for speed, but it means the model has **no built-in sense of order**. These two sentences use the same words but mean different things:

- "He painted the **car red**."
- "He painted the **red car**."

Without position information, the model would see them as the same.

**The solution.** We **add a position vector** to each word embedding. The vector for position `pos` is built from **sine and cosine waves** of different frequencies:

For even dimensions (`2i`), use sine:

$$
PE_{(pos,\,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

For odd dimensions (`2i+1`), use cosine:

$$
PE_{(pos,\,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

Here `pos` is the position of the word (0, 1, 2, ...), `i` numbers the pairs of dimensions, and $d_{model}$ is the embedding size.

> **An analogy: a clock.** To tell the time, a clock uses several hands that move at **different speeds**: seconds, minutes, hours. Together they make a unique "reading" for every moment. The sine and cosine waves in a positional encoding are like clock hands: some change quickly from one position to the next, some change slowly. Together they make a unique fingerprint for every position.

**Why this formula works:**

| Property | Why it helps |
|---|---|
| **Every position gets a unique pattern** | The model can tell the positions apart |
| **Values stay between −1 and 1** | Sine and cosine are periodic, so the numbers stay small, even for very long texts |
| **Different frequencies for different dimensions** | Some dimensions change quickly (fine, nearby positions) and others slowly (coarse, far-apart positions) |

**A tiny example** with $d_{model} = 4$ (each row is the encoding of one position):

| Position | dim 0 (sin) | dim 1 (cos) | dim 2 (sin) | dim 3 (cos) |
|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.0100 | 1.0000 |
| 2 | 0.9093 | −0.4161 | 0.0200 | 0.9998 |
| 3 | 0.1411 | −0.9900 | 0.0300 | 0.9996 |

**Three kinds of positional encoding exist:**

| Kind | How it works | Used in |
|---|---|---|
| **Sinusoidal (fixed)** | Sine and cosine waves; nothing to learn | The **original Transformer**; the models in these labs |
| **Learned (trainable)** | A position vector for every position, learned during training | **BERT** and **GPT** |
| **Rotary (RoPE)** | Rotates the query and key vectors according to position | Many recent LLMs, such as LLaMA |

> **Careful:** an earlier version said sinusoidal encoding is "used in BERT". That is **not** correct: BERT uses **learned** position embeddings. The BERT code in this manual uses the sinusoidal class for simplicity.

**The code.** This is the version used in the GPT, BERT, and translation labs. It adds the positions to a tensor of shape **(sequence length, batch, embedding size)**:

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

> **Note:** this `forward` **adds** the position to its input and applies dropout. Remember this when we build the BERT embedding in Part 7.

**The model input** is the **sum** of the two pieces:

| Piece | It tells the model | Shape |
|---|---|---|
| **Word embedding** | **What** the word means | (sequence length, $d_{model}$) |
| **Positional encoding** | **Where** the word is | (sequence length, $d_{model}$) |
| **Model input** | What + where = embedding **plus** positional encoding (added, not joined) | (sequence length, $d_{model}$) |

### 2.5 Shapes and conventions

When many sentences are processed together, we need a **batch** dimension. Two conventions exist:

| Convention | Shape | Set with |
|---|---|---|
| **Sequence-first** (GPT, BERT, and translation labs) | (sequence length, batch, features) | the default (`batch_first=False`) |
| **Batch-first** (text classifier lab) | (batch, sequence length, features) | `batch_first=True` |

**Example:** 32 sentences of 10 tokens each, with embedding size 64:

| Stage | Sequence-first shape | Batch-first shape |
|---|---|---|
| Token IDs | (10, 32) | (32, 10) |
| After the embedding | (10, 32, 64) | (32, 10, 64) |

Mixing the two conventions silently mixes up the dimensions, so keep one convention in each model.

---

## Part 3: Attention, the Heart of the Transformer

Now each word has a meaning (embedding) and a position. How does the model understand **context**, for example which words relate to which? The answer is **self-attention**.

### 3.1 The idea in plain words

Look at this sentence:

> "The animal didn't cross the street because **it** was too tired."

What does "it" mean? The animal, not the street. We know this because, when we read "it", our mind **looks back** at the other words and decides which ones matter. **Self-attention** does the same: while the model processes one word, it **looks at all the other words** and decides **how much attention** to pay to each. Then it mixes the useful information into that word.

> **An analogy: a highlighter.** For every word, the model reads the whole sentence with a highlighter. Words that help to understand this word get highlighted strongly. Words that do not help stay faint.

### 3.2 A warm-up: a dictionary made of matrices

The attention lab starts with a **translation dictionary** written as matrices. It shows the whole idea with no hard math.

- **K (keys):** each row is the one-hot vector of a **French** word.
- **V (values):** each row is the one-hot vector of the matching **English** word.
- **Q (query):** the one-hot vector of the French word that we want to translate.

The translation is **`Q @ K.T @ V`**:

1. `Q @ K.T` gives a vector with a **1 at the position of the matching row** (and 0 elsewhere). This is the dictionary lookup: "which entry matches my word?"
2. Multiplying by `V` **picks that row**: the English word.

This is a **hard lookup**: it only works when the query is exactly one of the keys. Real language is fuzzy, so the lab adds a **softmax**:

$$
\text{Output} = \text{softmax}(Q \cdot K^T) \cdot V
$$

Now a query that is *similar* (but not identical) to some keys returns a **blend** of their values, weighted by how well they match. In a small check, a query with a large score for the second key gave the weights `[0.007, 0.987, 0.007]`: almost a hard lookup, but still smooth. This **soft lookup** is exactly what attention does.

### 3.3 Query, Key, and Value (Q, K, V)

In a real Transformer, every token makes three vectors from its embedding. Each one is the embedding multiplied by its own **learned weight matrix** ($W_Q$, $W_K$, $W_V$):

| Vector | The question it answers | A library analogy |
|---|---|---|
| **Query (Q)** | What am I looking for? | The question we ask the librarian |
| **Key (K)** | What do I have? | The label on a book's spine |
| **Value (V)** | What information do I carry? | The content inside the book |

We compare our question (query) with every label (key). Books whose labels match well are the ones whose contents (values) we read most.

### 3.4 The process, step by step

1. Compare the **query** of word $i$ with the **key** of every word $j$, using the **dot product** (Part 0.2). A bigger score means a better match.
2. **Scale** the scores, then apply **softmax**. This gives the **attention weights**: positive numbers that add up to 1.
3. The weights decide how much attention word $i$ pays to each word. A higher weight means the value $V_j$ matters more.
4. The output for word $i$ is the **weighted sum of all the value vectors**.

### 3.5 Scaled dot-product attention

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

And what a large spread does to softmax: the scores `[8, 0, -8]` give the weights `[0.9997, 0.0003, 0.0000]` (the model looks at only one word), while the scaled scores `[1, 0, -1]` give `[0.665, 0.245, 0.090]` (a smoother, more useful mix).

### 3.6 A worked example

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

Each token's output is now a **mix of all the tokens' values**, with more weight on the tokens that match its query best.

### 3.7 Attention in code

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

### 3.8 Multi-head attention: several highlighters

One attention calculation can only look for one kind of relationship at a time. So Transformers run **several in parallel**. Each one is called a **head**.

> **An analogy.** Several readers read the same sentence, each with a different colored highlighter: one marks grammar, one marks meaning, one marks who-did-what. At the end, we combine all their notes. (That different heads learn different things is a common observation, not a guaranteed rule.)

- **The idea:** the embedding size $d_{model}$ is shared among $h$ heads. Each head works on a smaller part of size $d_k = d_{model} / h$, with its own $W_Q$, $W_K$, $W_V$.
- **The process:**
  1. Project the input into $h$ sets of Q, K, and V.
  2. Compute scaled dot-product attention in **each head separately**.
  3. **Concatenate** (join) the outputs of all heads.
  4. A final **linear layer** mixes the information from all the heads.

> **Careful:** $d_{model}$ must be **divisible** by the number of heads (for example, 64 with 8 heads gives $d_k = 8$). Otherwise PyTorch raises an error.

In PyTorch, `nn.MultiheadAttention` does all of this (and it is used inside `nn.TransformerEncoderLayer`):

```python
mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
out, attn_weights = mha(x, x, x)          # self-attention: the query, key, and value are all x
```

### 3.9 Masks: who is allowed to look at whom?

A **mask** controls what each token is allowed to "see". There are three ideas, and one of them is often confused with the other two:

| Mask | Purpose | Kind | Used in |
|---|---|---|---|
| **Causal mask** | Stops a token from attending to **future** tokens | Attention mask | Decoders (GPT, the translation decoder) |
| **Padding mask** | Ignores `<pad>` tokens | Attention mask | All models |
| **MLM masking** | Hides some input words that the model must predict | **Data corruption**, not an attention mask | BERT |

> **Careful:** an earlier version listed the "MLM mask" next to the attention masks. MLM masking is different: it **replaces words in the input** with `[MASK]` (Part 7). It does not block any attention.

#### The causal mask

When a model writes text, it must not peek at the words that come **after** the current one. That would be cheating: it would see the answer. The **causal mask** blocks attention to the future. It sets the scores of future positions to $-\infty$ before the softmax, so their weights become exactly 0.

With the three tokens from Section 3.6, the weights become:

| | token 1 | token 2 | token 3 |
|---|---:|---:|---:|
| **token 1** | 1.000 | 0 | 0 |
| **token 2** | 0.330 | 0.670 | 0 |
| **token 3** | 0.248 | 0.248 | 0.504 |

Token 1 can only see itself, token 2 sees tokens 1 and 2, and token 3 sees all three.

**The causal mask code** (each **row** is a token that is looking; `-inf` means "cannot look at this position"):

```python
def generate_square_subsequent_mask(sz, device=DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
```

A causal mask for 4 tokens looks like this:

```text
[[0, -inf, -inf, -inf],
 [0,   0,  -inf, -inf],
 [0,   0,    0,  -inf],
 [0,   0,    0,    0]]
```

> **Note on conventions:** this mask is an **additive float** mask (0 = allowed, −inf = blocked). PyTorch also accepts a **boolean** mask where `True` means *blocked*, for example the one below. Both do the same job. The built-in `nn.Transformer.generate_square_subsequent_mask(n)` makes the float version.

```python
n = 3
mask = torch.triu(torch.ones(n, n), diagonal=1).bool()     # True above the diagonal = the future
# or: nn.Transformer.generate_square_subsequent_mask(n)
```

#### The padding mask

When we put several sentences in a batch, the short ones are filled with `<pad>` tokens to reach the same length. Padding carries no meaning, so the **padding mask** tells attention to ignore those positions. We build it by checking where the token IDs equal the padding ID: `tokens == PAD_IDX`.

### 3.10 Part 3 summary

| Concept | One-sentence summary |
|---|---|
| **Self-attention** | Lets every token look at all the other tokens and mix in the useful information |
| **Query, key, value** | Queries are matched against keys, and the matching values are mixed together |
| **Scaled dot-product attention** | $\text{softmax}(QK^T/\sqrt{d_k})V$: a weighted sum of values, based on query-key similarity |
| **Why scale** | Keeps the scores in a healthy range, so softmax does not saturate |
| **Multi-head attention** | Several attention calculations in parallel, to capture different relationships |
| **Causal mask** | Hides the future (used when generating text) |
| **Padding mask** | Hides the padding tokens |

---

## Part 4: Inside a Transformer

We know the two main ingredients: **embeddings with positions** (Part 2) and **attention** (Part 3). Now we see how they are assembled into layers, and how the layers form encoders and decoders.

### 4.1 Inside one encoder layer

A Transformer layer repeats the same small recipe. PyTorch's `nn.TransformerEncoderLayer` is exactly this recipe:

| Step | What happens | In simple words |
|---:|---|---|
| 1 | **Multi-head self-attention** | Every word **listens** to the other words |
| 2 | **Add & Norm** | Keep the original information, and tidy up the numbers |
| 3 | **Feed-forward network** | Every word **thinks** about what it heard, on its own |
| 4 | **Add & Norm** again | Keep and tidy again |

```text
x → [ self-attention ] → add x → layer norm → [ feed-forward ] → add → layer norm → output
```

Let us explain the three new pieces:

- **Residual connection ("Add").** We **add the layer's input back** to its output. It works like a shortcut: the original information is never lost, even if the layer makes a mistake. It also helps the gradients flow during training, which makes deep stacks of layers trainable.
- **Layer normalization ("Norm").** It rescales the numbers of each token so that they stay on a similar scale, a bit like converting exam scores from different tests to the same scale. This keeps training stable.
- **Feed-forward network.** Two linear layers with an activation (such as ReLU) in between. It is applied to **each token separately**. Attention mixes information **between** words, and the feed-forward network processes the information **inside** each word.

| Setting of `nn.TransformerEncoderLayer` | Meaning |
|---|---|
| `d_model` | The embedding size (the size of each token's vector) |
| `nhead` | The number of attention heads (`d_model` must be divisible by it) |
| `dim_feedforward` | The hidden size of the feed-forward network |
| `dropout` | How much random "switching off" is used against overfitting |
| `batch_first` | `True` for (batch, sequence, features); `False` (default) for (sequence, batch, features) |

`nn.TransformerEncoder` **stacks several of these layers**: the output of each layer is the input of the next one.

### 4.2 Stacking layers

A real model stacks many layers (for example 12 layers in BERT-base). Each layer lets the words exchange information once more. A common observation is that the first layers capture simple patterns (such as nearby words), and later layers capture more abstract relationships. The **size** of a model is described by three numbers: how many layers, how many heads, and the embedding size.

| Model | Layers | Heads | Embedding size |
|---|---:|---:|---:|
| BERT-base (the defaults in the BERT code of Part 7) | 12 | 12 | 768 |

### 4.3 Encoder and decoder

The original Transformer was designed for translation and has **two parts**:

| | **Encoder** | **Decoder** |
|---|---|---|
| **Job** | Read and understand the input sequence | Generate an output sequence, one token at a time |
| **Self-attention** | **Unmasked** (bi-directional): each token can look at all the others, before and after it | **Masked**: each token can look **only at earlier tokens** |
| **Why** | To build a rich, contextual representation of the whole input | The model must not see the answer it is trying to predict |
| **Extra attention** | None | **Encoder-decoder attention** (cross-attention): looks at the encoder's output to stay on topic |
| **PyTorch** | `nn.TransformerEncoderLayer`, `nn.TransformerEncoder` | `nn.TransformerDecoderLayer`, `nn.TransformerDecoder` |
| **Output** | A sequence of context-aware embeddings | The next-token predictions |

**Cross-attention in one sentence:** the decoder asks questions (queries), and the encoder's output provides the labels and contents (keys and values), so the decoder can look back at the input while it writes the output.

> **Careful:** modern models often use only one of the two parts. The decoder in the table (with cross-attention) is the **original** one. A **GPT-style** model has no encoder, so it has no cross-attention: it is a stack of masked self-attention layers only.

| Family | Example | Attention | Good for |
|---|---|---|---|
| **Encoder-only** | BERT | Unmasked (sees the whole input) | Understanding: classification, tagging |
| **Decoder-only** | GPT | Masked (sees only the past) | Generating text |
| **Encoder-decoder** | The original Transformer, T5 | Encoder unmasked; decoder masked plus cross-attention | Translation, summarization |

### 4.4 What sits on top of the Transformer

The Transformer layers produce one vector for every token. To get an **answer**, we add a small **head** on top. The head depends on the task:

| Task | Head on top | What it outputs |
|---|---|---|
| **Generate the next word** (GPT) | A linear layer (`lm_head`) from the embedding size to the vocabulary size | One score for every word in the vocabulary |
| **Guess a hidden word** (BERT, MLM) | A linear layer to the vocabulary size, for every position | A word prediction for every position |
| **Do two sentences follow each other?** (BERT, NSP) | A linear layer to 2 classes, on the `[CLS]` output | Two scores |
| **Classify a whole text** | Pooling (average of all tokens, or the `[CLS]` vector), then a linear layer | One score per class |
| **Translate** | A linear layer from the decoder output to the target vocabulary | One score for every target word |

### 4.5 The main PyTorch modules

| Module | What it is |
|---|---|
| `nn.Embedding` | Lookup table: token ID → vector |
| `nn.MultiheadAttention` | Multi-head attention |
| `nn.TransformerEncoderLayer` | One encoder layer: self-attention + feed-forward network (with residuals and layer norm) |
| `nn.TransformerEncoder` | A stack of encoder layers |
| `nn.TransformerDecoderLayer` | One decoder layer (masked self-attention, cross-attention, feed-forward) |
| `nn.TransformerDecoder` | A stack of decoder layers |
| `nn.Transformer` | A full encoder-decoder Transformer (both stacks) |

---

## Part 5: Training and Generating

### 5.1 What do we train the model to do?

The great trick of language models: **the text itself provides the answers**. Nobody has to label the data by hand. We simply hide part of the text and ask the model to guess it. This is called **self-supervised learning**.

| Family | Training task | Example |
|---|---|---|
| **Decoder-only** | Predict the **next** token | `The cat sat on the` → `mat` |
| **Encoder-only** | Predict **hidden** tokens (and sentence order) | `The [MASK] sat on the mat` → `cat` |
| **Encoder-decoder** | Predict the next **target** token, given the source sentence | `Ein Hund ...` → `A dog ...` |

After this **pre-training** on a huge amount of text, a model can be **fine-tuned**: trained a little more on a small, specific task (for example, sentiment analysis).

### 5.2 Loss functions

| Task | Loss | Note |
|---|---|---|
| **Next-token prediction** (language modeling) | Cross-entropy over the vocabulary | GPT, the translation decoder |
| **Masked-token prediction** (MLM) | Cross-entropy over the vocabulary | BERT |
| **Next-sentence prediction** (NSP) | Cross-entropy over 2 classes | BERT |
| **Classification** | Cross-entropy over the classes | The text classifier of Part 8 |

When we compute the loss, we **ignore the padding tokens** with `ignore_index=PAD_IDX`, because padding has nothing to predict. PyTorch's `nn.CrossEntropyLoss` takes the raw scores (**logits**) and applies the softmax itself, so we do not apply softmax first.

### 5.3 The training loop

Every model in this manual is trained with the same loop (Part 0.4):

| Step | What we do |
|---:|---|
| 1 | Take a **batch** of examples |
| 2 | **Forward pass:** the model makes predictions (logits) |
| 3 | **Loss:** measure how wrong the predictions are |
| 4 | **`zero_grad()`:** clear the old gradients |
| 5 | **`backward()`:** compute the new gradients |
| 6 | **Clip** the gradients (optional, keeps them from becoming too large) |
| 7 | **`step()`:** update the weights |
| 8 | Repeat for all batches (one **epoch**), then for many epochs |

```python
model.train()                                              # switch on dropout
for src, tgt in train_data:
    logits = model(src)                                    # 2. forward pass
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                   tgt.reshape(-1))                        # 3. loss (flatten the positions)
    optimizer.zero_grad()                                  # 4. clear old gradients
    loss.backward()                                        # 5. compute new gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 6. gradient clipping
    optimizer.step()                                       # 7. update the weights
```

### 5.4 Teacher forcing and the one-step shift

While training a model that writes text, we know the correct answer. Instead of letting the model feed on its own (possibly wrong) guesses, we **give it the correct previous words**. This is **teacher forcing**. We build the input and the answer by **shifting the text by one position**:

```text
Text:       [<bos>, The, cat, sat, on, the, mat, <eos>]

Input:      [<bos>, The, cat, sat, on, the, mat]        ← the text without the last token
Answer:     [The,   cat, sat, on,  the, mat, <eos>]     ← the text without the first token
```

At every position, the model must predict the **next** token. Because all positions are given at once (and the causal mask hides the future), the whole sentence is trained **in parallel**. This is much faster than an RNN.

At **translation or generation time**, there is no correct answer to give. The model must use its **own** previous predictions. That is the difference between training and generating.

### 5.5 Training big models: the optimization toolbox

Training large Transformers needs a lot of **memory** and **computing time**. The course introduces several techniques to cope.

| Technique | The problem | The idea | PyTorch tool |
|---|---|---|---|
| **Gradient accumulation** | The batch we want does not fit in GPU memory | Add up the gradients of several small batches, then update once | Several `loss.backward()` calls before one `optimizer.step()` |
| **Mixed-precision training** | 32-bit numbers (FP32) use a lot of memory and are slow | Use 16-bit numbers (FP16 or BF16) for most calculations | `torch.autocast` and `GradScaler` (AMP) |
| **Distributed training** | The model or the data is too big for one machine | Use several GPUs or machines | `DistributedDataParallel` |
| **AdamW** | Plain Adam applies weight decay in a less effective way | Apply weight decay separately from the gradient update | `torch.optim.AdamW` |
| **LAMB** | Very large batches make training unstable | Adapt the learning rate for each layer | A library (LAMB is not in core PyTorch) |

#### Gradient accumulation

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

#### Mixed-precision training (AMP)

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

#### Distributed training

| Type | How it works | When to use it |
|---|---|---|
| **Data parallelism** | Every GPU holds a **copy of the whole model** and gets a **part of the data**. The gradients are averaged across the GPUs, so all copies stay the same | The model fits on one GPU, but we want to train faster |
| **Model parallelism** | The **model itself is split** across GPUs (for example, different layers on different GPUs) | The model is too big for a single GPU |

In PyTorch, data parallelism is usually done with `DistributedDataParallel` (and a `DistributedSampler` to give each GPU its own part of the data):

```python
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model.to(device), device_ids=[local_rank])
```

#### Efficient optimizers: AdamW and LAMB

- **AdamW:** a variant of Adam that fixes how **weight decay** (a regularization technique that discourages large weights and helps against overfitting) is applied. It is applied directly to the weights, separately from the adaptive gradient update. This usually gives **better generalization**, and it is a very common choice for Transformers.
- **LAMB:** designed for training with **very large batch sizes**. It adapts the learning rate **layer by layer**, which often means less tuning when the batch size grows.

Two more habits are common in Transformer training: a learning-rate **warm-up** (start small and grow the learning rate for the first steps) and **gradient clipping** (`torch.nn.utils.clip_grad_norm_`).

### 5.6 Generating text

A trained language model writes **one token at a time**. This is called **autoregressive** generation: each new token is predicted from the tokens before it, then added to the text, and the loop repeats.

```text
prompt → model → scores for the next token → choose one → append it → repeat until <eos>
```

How do we **choose** the next token from the scores? There are several strategies:

| Strategy | How it chooses | Effect |
|---|---|---|
| **Greedy decoding** | Always the most probable token | Predictable, but can become repetitive |
| **Sampling** | Draws a token at random, according to the probabilities | More varied |
| **Temperature** | Divides the scores by $T$ before the softmax | $T<1$ makes the choice safer; $T>1$ makes it more varied |
| **Top-k** | Chooses only among the $k$ most likely tokens | Cuts the unlikely words |
| **Top-p** | Chooses among the smallest group of tokens whose probabilities add up to $p$ | Adapts the group size to the situation |

**A small example.** The scores (logits) are `[2.0, 1.0, 0.1]` for three words:

| Setting | Probabilities |
|---|---|
| Temperature 1 (normal) | 0.659, 0.242, 0.099 |
| Temperature 0.5 (safer) | 0.864, 0.117, 0.019 |
| Temperature 2 (more varied) | 0.502, 0.304, 0.194 |
| Temperature 1, then top-k = 2 | 0.731, 0.269, 0 (the third word is removed and the rest are renormalized) |
| Temperature 1, then top-p = 0.9 | The first two words are kept, because 0.659 + 0.242 = 0.901 reaches 0.9 |

### 5.7 How do we know the model is good?

| Score | What it tells us | Used for |
|---|---|---|
| **Loss** | How wrong the model is (lower is better) | Training any model |
| **Accuracy** | How often the top guess is exactly right | Classification |
| **Perplexity** | $e^{\text{loss}}$: how uncertain the model is about the next word. A perplexity of 8 means it is about as unsure as choosing among 8 equally likely words | Language models |
| **BLEU** | How close a translation is to reference translations | Translation (Part 9) |

### 5.8 Part 5 summary

| Concept | One-sentence summary |
|---|---|
| **Self-supervised learning** | The text itself gives the answers (the next word, or a hidden word) |
| **Cross-entropy loss** | Punishes a low probability on the right answer; padding is ignored |
| **Training loop** | Forward, loss, `zero_grad`, `backward`, clip, `step` |
| **Teacher forcing** | Give the correct previous words while training; shift the text by one position |
| **Optimization toolbox** | Gradient accumulation, mixed precision, distributed training, AdamW or LAMB |
| **Generation** | One token at a time; greedy, sampling, temperature, top-k, top-p |

---

## Part 6: Decoder-Only Models (GPT-like)

> **In simple words:** a decoder-only model is a very good **autocomplete**. It reads the text so far and predicts the next token, again and again, and that is how it writes stories, answers, and code.

### 6.1 Core principle

Decoder-only models are trained with **causal language modeling (CLM)**: *predict the next token, given all the tokens before it.*

```text
Given:    [The, cat, sat, on, the]
Predict:  [cat, sat, on, the, mat]
```

At every position, the model must predict the next token. It can only attend to **past and present** tokens, never to future ones. This is enforced by the **causal mask** (Part 3.9).

### 6.2 Architecture

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

### 6.3 The model code (`CustomGPTModel`)

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

### 6.4 Masking in decoder-only models

**The causal mask is the defining feature.** Without it, the model could "cheat" during training by looking at the answer.

```python
def create_mask(src, device=DEVICE):
    src_seq_len = src.shape[0]
    src_mask = generate_square_subsequent_mask(src_seq_len)     # triangular causal mask
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)         # True where the token is padding
    return src_mask, src_padding_mask
```

Here `src` must contain **token IDs** (not embeddings), so that the padding positions can be found.

### 6.5 Data preparation

The `get_sample` function cuts a random piece of text (a **block**) and creates an input and a target from it:

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

**Key insight:** `tgt_sequence` is `src_sequence` **shifted by one position**. That shift turns the data into a next-token prediction task. For example, with `block_size = 4` and the text `[the, cat, sat, on, the, mat]`, if the random start is 0:

| | Tokens |
|---|---|
| `src_sequence` (the input) | the, cat, sat, on |
| `tgt_sequence` (the answers) | cat, sat, on, the |

### 6.6 Training

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

The `logits` have the shape (sequence, batch, vocabulary) and the targets have the shape (sequence, batch). `reshape(-1, vocab)` flattens all positions into one long list of predictions, so the loss can compare each prediction with its target token. Padding tokens are ignored by the loss.

### 6.7 Inference: autoregressive generation

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

> **Note:** greedy decoding always takes the top token, so the text can become repetitive. Sampling methods (**temperature**, **top-k**, **top-p**) add controlled randomness (Part 5.6).

### 6.8 Using a pre-trained GPT-2

Training a GPT from scratch needs huge data and computers. In practice, we often **download a pre-trained model**:

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

> **Careful:** the first version left out `do_sample=True`. Without it, `generate` uses greedy decoding and **ignores** `temperature`, `top_k`, and `top_p` (the library prints a warning). Also, `max_length` would count the prompt tokens too, so we use `max_new_tokens`.

### 6.9 Summary: decoder-only models

| Aspect | Details |
|---|---|
| **Attention** | Causal (left to right only) |
| **Training objective** | Next-token prediction (causal language modeling) |
| **Key mask** | Causal (triangular) |
| **Special tokens** | `<bos>`, `<eos>`, `<pad>`, `<unk>` (GPT-2 uses `<\|endoftext\|>` for the start and the end) |
| **Positional encoding** | Learned in GPT models; sinusoidal in this lab's custom model |
| **Use cases** | Text generation, chatbots, code generation |

---

## Part 7: Encoder-Only Models (BERT-like)

> **In simple words:** an encoder-only model is a very good **reader**. It does not write text. It reads a whole text, looking in **both directions**, and builds a deep understanding of every word in its context.

### 7.1 Core principle

Encoder-only models build a **deep, bidirectional understanding** of text. They look at the whole sequence at once, with **no causal mask**, and create a rich, contextual embedding for every token.

Why both directions? Take the sentence "The ___ sat on the mat". To guess the missing word, it helps to see the words on **both sides**: "sat" (after) and "on the mat" (also after) tell us it is probably an animal, and "The" (before) says it is a noun. A model that sees only the left side is at a disadvantage here.

BERT is pre-trained with **two objectives**:

1. **Masked language modeling (MLM):** predict randomly hidden tokens, using context from **both** directions.
2. **Next sentence prediction (NSP):** predict whether sentence B really follows sentence A.

> **Note:** later models dropped NSP. **RoBERTa**, for example, is trained with MLM only, because NSP was found not to help much. We cover NSP because the original BERT and this lab use it.

### 7.2 Architecture

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

### 7.3 Task 1: masked language modeling (MLM)

**Goal:** hide some tokens and predict them from the context on both sides. It is like a fill-in-the-blank exercise.

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

### 7.4 Task 2: next sentence prediction (NSP)

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

### 7.5 The three BERT embeddings

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

> **Careful:** the first version computed `token_emb + self.positional_encoding(token_emb) + segment`, which equals `2 × token_emb + position + segment`. Because `PositionalEncoding.forward` already adds its input (Part 2.4), the token embedding must be added only once.

### 7.6 The data preparation pipeline

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

**Stage 4 code** (it pads each pair to the same length, converts tokens to IDs, and builds the segment IDs: 1 for sentence A, 2 for sentence B, 0 for padding):

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

### 7.7 The BERT model

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

### 7.8 Training: a combined loss

BERT is trained on **both** tasks at the same time. The total loss is the **sum** of the two losses:

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

### 7.9 Using a pre-trained BERT for inference

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

### 7.10 Summary: encoder-only models

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

## Part 8: Hands-On: A Transformer Text Classifier

> **Lab for this part:** M3-L2, Applying Transformers for Classification.

In this part we build a complete, small Transformer that **reads a news text and decides its topic**. It is an **encoder-only** model with a classification head on top.

### 8.1 The task

The dataset is **AG_NEWS**: short news items with four topics.

| Label | Topic |
|---:|---|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tec |

### 8.2 The recipe

```text
text → tokens → IDs → embedding → + positional encoding
     → Transformer encoder layers → average over the tokens (pooling)
     → linear layer → 4 scores (logits) → class
```

The pieces are the ones we already know. The embedding is shown in Part 2.3, and the encoder layers in Part 4.1.

### 8.3 Batch-first positional encoding

This lab uses **batch-first** tensors (batch, sequence, features), so it has its own version of the positional encoding. To avoid mixing it up with the sequence-first class of Part 2.4, we call it `BatchFirstPositionalEncoding` here.

```python
class BatchFirstPositionalEncoding(nn.Module):
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

> **Note:** the lab's code named the second argument `vocab_size`, but it is really the **maximum sequence length** (`max_len`), because it is the number of positions in the table. We use the clearer name here. Also, this code needs an **even** `d_model` (the sine and cosine halves must have the same size), and sequences **longer than `max_len`** cause an error.
>
> **Check:** `div_term` is just a faster way to write $1/10000^{2i/d_{model}}$. We checked numerically that the vectorized code gives exactly the same table as the formula.

### 8.4 The classifier model

```python
class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, dim_feedforward, num_layers, num_class, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = BatchFirstPositionalEncoding(embedding_dim, dropout=dropout)

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

> **Note:** the lab's constructor arguments are not shown in the original notes, so we wrote typical ones. The important detail is `batch_first=True`. By default, `nn.TransformerEncoderLayer` expects the shape (sequence, batch, features). Our positional encoding and `x.mean(dim=1)` assume (batch, sequence, features), so the two must match.

#### Pooling and the final linear layer

After the Transformer layers, we have **one vector for every token**, but a classification task needs **one prediction for the whole text**:

- **Pooling:** `x.mean(dim=1)` averages the vectors of all tokens into a single vector for the whole text.
- **Linear layer:** `self.classifier` projects that vector from the hidden size (for example 64) to the number of classes (4 for AG_NEWS). The result is the **logits**, which go to `CrossEntropyLoss`.

> **Two improvements to know about.** (1) If the batch contains **padding**, the mean also averages the padding tokens. Passing a padding mask (`src_key_padding_mask`) and averaging only the real tokens is more accurate. (2) Instead of the mean, BERT-style models add a special **[CLS]** token and classify from its output vector.

### 8.5 The training process in the lab

The lab gives a standard training loop for text classification. It does not use the optimization techniques of Part 5.5, but this is where they would be added.

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

### 8.6 Part 8 summary

| Piece | What it does |
|---|---|
| **Embedding + positional encoding** | Words and their order as numbers |
| **`nn.TransformerEncoder`** | Lets the words exchange information |
| **Mean pooling** | One vector for the whole text |
| **Linear layer** | One score per topic |
| **`CrossEntropyLoss`** | Trains the model |

---

## Part 9: Encoder-Decoder Models (Translation)

> **In simple words:** an encoder-decoder model is like a **human interpreter**. The encoder **listens to the whole sentence** in the source language and understands it. The decoder then **speaks the translation**, word by word, always keeping the original meaning in mind.

### 9.1 Core principle

Encoder-decoder models map an **input sequence** to an **output sequence**:

- an **encoder** reads and understands the whole source sentence;
- a **decoder** writes the target sentence word by word, guided by the encoder's output.

The classic use case is **machine translation**, for example German → English.

### 9.2 Architecture

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

### 9.3 The three attention mechanisms

| Attention | Queries (Q) | Keys (K) and values (V) | Purpose |
|---|---|---|---|
| **Encoder self-attention** | Source tokens | Source tokens | Understand the source sentence |
| **Decoder masked self-attention** | Target tokens | Past target tokens only | Generate the target sequence |
| **Cross-attention** | Target tokens | Source tokens (the memory) | Look at the source while generating |

**Cross-attention is the bridge** between input and output: at every decoding step, it calculates how much attention to pay to each position of the source sentence.

### 9.4 The `Seq2SeqTransformer` model

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

### 9.5 Masking in encoder-decoder models

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

### 9.6 Teacher forcing

During training, the decoder receives the **correct** target words, not its own predictions (Part 5.4):

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

### 9.7 The training loop

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

### 9.8 Inference: greedy decoding

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

### 9.9 The translation function

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

### 9.10 Evaluating with the BLEU score

The **BLEU score** (Bilingual Evaluation Understudy) measures translation quality by comparing the generated text with one or more **reference translations**. Using several references is useful, because a sentence often has more than one good translation.

BLEU has three parts:

| Part | What it does |
|---|---|
| **N-gram precision** | Counts how many 1-word, 2-word, 3-word, and 4-word groups of the output also appear in a reference. Each group is counted **at most as many times as it appears in the reference** ("clipping"), so repeating a word does not help |
| **Brevity penalty** | Punishes outputs that are too short. If the output length $c$ is longer than the reference length $r$, the penalty is 1. Otherwise it is $e^{1 - r/c}$ |
| **Geometric mean** | Combines the four precisions (usually with equal weights) and multiplies by the brevity penalty |

BLEU is between 0 and 1, and **higher is better**.

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

> **Note:** BLEU compares exact tokens, so the text must have the same case and tokenization. We lowercase both sides and add **smoothing**; without it, a short sentence with no matching four-word group scores almost 0. As an example computed with NLTK, the output `an asian man is sweeping the walkway .` scores about **0.95** against two reference translations, while the two-word output `man walking` scores about **0** without smoothing and **0.02** with smoothing. Note also that `"In the snow, a brown dog..."` has a comma attached to the word "snow,", so it never matches the token "snow" exactly. Tokenize the punctuation consistently.

### 9.11 A practical application: translating a PDF

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

### 9.12 Summary: encoder-decoder models

| Aspect | Details |
|---|---|
| **Attention** | Encoder self-attention + decoder masked self-attention + cross-attention |
| **Training** | Teacher forcing (the correct target tokens are the decoder input) |
| **Inference** | Autoregressive greedy decoding |
| **Masks** | Causal (target), padding (source and target) |
| **Special tokens** | `<bos>`, `<eos>`, `<pad>`, `<unk>` |
| **Use cases** | Translation, summarization, dialogue systems, code-to-text |

---

## Part 10: Choosing the Right Family

### 10.1 Side-by-side comparison

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

### 10.2 Decision guide

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

### 10.3 What all three families share

- **Attention** (Q, K, V) connects the tokens.
- **Positional encoding** preserves the word order.
- **Masking** controls what each token can see.
- **Cross-entropy loss** drives the learning.
- **Autoregressive generation** produces the output one token at a time (in the generating models).

The families differ only in **which attention they use** (causal, bidirectional, or both plus cross-attention) and **what they are trained to predict**.

---

## Part 11: Reference

### 11.1 The big ideas in twelve lines

1. A **neural network** learns by guessing, measuring the error (loss), and nudging its weights.
2. Text is cut into **tokens**, and each token gets an **ID**.
3. An **embedding** turns an ID into a vector of numbers that captures meaning.
4. A **positional encoding** is added, because Transformers read all words at once and would otherwise ignore the order.
5. **Self-attention** lets every word look at every other word and mix in the useful information.
6. Attention uses **queries, keys, and values**: $\text{softmax}(QK^T/\sqrt{d_k})V$.
7. **Multi-head** attention runs several attentions in parallel.
8. A **layer** = attention + add & norm + feed-forward + add & norm. Many layers are stacked.
9. **Masks** decide who may look at whom: the **causal mask** hides the future.
10. **Decoder-only** models generate, **encoder-only** models understand, **encoder-decoder** models translate.
11. Models are trained to **predict text** (next word or hidden word), with **cross-entropy** loss, and with tricks (accumulation, mixed precision, AdamW) for big sizes.
12. Text is **generated** one token at a time, by choosing from probabilities (greedy, temperature, top-k, top-p).

### 11.2 Glossary

| Term | Definition |
|---|---|
| **Add & Norm** | A residual connection (add the input back) followed by layer normalization |
| **AdamW** | An optimizer that applies weight decay separately from the gradient update |
| **Attention** | A mechanism that lets each token weight the importance of other tokens |
| **Attention head** | One of several parallel attention calculations in multi-head attention |
| **Autoregressive** | Generating the output one token at a time, using the previous outputs as input |
| **Batch** | A small group of examples processed together |
| **BLEU score** | A metric for translation quality, comparing to reference translations |
| **Causal language modeling** | Predicting the next token from the previous tokens only |
| **Causal mask** | Prevents a token from attending to future tokens |
| **Cross-attention** | Attention from the decoder to the encoder output (in sequence-to-sequence models) |
| **Cross-entropy loss** | A loss that compares the predicted probabilities with the true labels |
| **$d_{model}$** | The embedding size: the length of each token's vector |
| **Decoder** | The component that generates the output sequence |
| **Embedding** | A learned vector of numbers that stands for a token |
| **Encoder** | The component that turns the input sequence into a representation |
| **Epoch** | One full pass through the training data |
| **Feed-forward network** | Two linear layers with an activation, applied to each token separately |
| **Fine-tuning** | Taking a pre-trained model and training it further on a specific task |
| **Gradient accumulation** | Adding up the gradients of several small batches before one update |
| **Greedy decoding** | Always choosing the most probable next token |
| **Head (output head)** | A small layer on top of the Transformer that produces the answer for a task |
| **Layer normalization** | Rescaling the numbers of each token to keep training stable |
| **Logits** | Raw output scores, before softmax |
| **Mixed precision** | Training with 16-bit numbers for most calculations, to save memory and time |
| **MLM (masked language modeling)** | Pre-training task: predict randomly hidden tokens |
| **NSP (next sentence prediction)** | Pre-training task: predict whether sentence B follows sentence A |
| **Padding** | Adding `<pad>` tokens to make sequences the same length |
| **Perplexity** | $e^{\text{loss}}$; lower means a more confident language model |
| **Positional encoding** | Information about token order, added to the embeddings (sinusoidal, learned, or rotary) |
| **Pre-training** | Training on large unlabeled data to learn general language patterns |
| **Q, K, V** | Query, key, value: the vectors used to compute attention |
| **Residual connection** | Adding a layer's input to its output, as a shortcut |
| **Segment embedding** | A vector that says which sentence (A or B) a token belongs to |
| **Self-attention** | Attention within one sequence: tokens attend to each other |
| **Self-supervised learning** | Learning from labels that come from the text itself |
| **Softmax** | Turns raw scores into probabilities that add up to 1 |
| **Teacher forcing** | Giving the decoder the correct tokens during training |
| **Temperature** | A number that makes the choice of the next token safer (below 1) or more varied (above 1) |
| **Token** | The basic unit of text (word, subword, or character) |
| **Tokenization** | Splitting text into tokens |
| **Top-k / top-p** | Sampling only from the k most likely tokens, or from the smallest group whose probabilities add up to p |
| **Transformer** | A neural network architecture based on attention |
| **Vocabulary** | The set of known tokens, mapped to integer indices |

### 11.3 Cheat sheet

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
| **Text classifier** | Classification | Predict the class of a whole text |

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

> **Careful:** the first version drew the decoder-only matrix as an **upper** triangle, which reads as "token 1 sees everything and token 4 sees only itself". That is the opposite of causal attention and of the mask in Part 3.9.

### 11.4 Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Forgetting the positional encoding | The model cannot tell the word order | Add it to the embeddings (Part 2.4) |
| Forgetting the causal mask in a decoder | The model "cheats" by seeing the next tokens | Always pass the causal mask (Part 3.9) |
| `batch_first` does not match the rest of the code | Attention mixes the wrong dimensions | Use one convention everywhere (Part 2.5) |
| $d_{model}$ not divisible by the number of heads | PyTorch raises an error | Choose compatible sizes (for example 64 and 8) |
| Sequence longer than `max_len` | The positional table is too short | Increase `max_len` or truncate the text |
| Adding the token embedding twice (in the BERT embedding) | The embeddings have the wrong scale | `PositionalEncoding` already adds its input (Part 7.5) |
| Confusing the NSP label convention | The answers are inverted | Hugging Face: 0 = continuation, 1 = random (Part 7.9) |
| Using `temperature`, `top_k`, or `top_p` without `do_sample=True` | The settings are silently ignored | Set `do_sample=True` (Part 6.8) |
| Not ignoring padding in the loss | The model is trained to predict `<pad>` | `ignore_index=PAD_IDX` |
| Averaging over padding tokens (in pooling) | The padding dilutes the pooled vector | Use a padding mask (Part 8.4) |
| Applying teacher forcing at inference | There is no correct target at translation time | Decode token by token (Part 9.8) |
| Computing BLEU without lowercasing or smoothing | Scores near 0 for short sentences | Make the text consistent and use smoothing (Part 9.10) |
| Forgetting `model.eval()` at inference | Dropout stays on, so predictions are noisy | Call `model.eval()` and use `torch.no_grad()` |
| Creating a scheduler but never calling `scheduler.step()` | The learning rate never changes | Call it once per epoch (Part 8.5) |

### 11.5 Self-check

Try each question first, then open the answer.

**1. What is a neural network, in one sentence?**

<details>
<summary>Show answer</summary>

A big math function with adjustable numbers (weights) that learns by guessing, measuring the error, and nudging the weights.

</details>

**2. What does softmax do?**

<details>
<summary>Show answer</summary>

It turns raw scores (logits) into probabilities that are positive and add up to 1. The largest score gets the largest probability.

</details>

**3. What was the problem with RNNs that Transformers solved?**

<details>
<summary>Show answer</summary>

RNNs read one word at a time, so they are slow (no parallel processing) and their memory fades over long texts. Transformers process the whole sequence at once and let every word look at every other word.

</details>

**4. Why do we use embeddings instead of plain word IDs?**

<details>
<summary>Show answer</summary>

An ID carries no meaning. An embedding is a learned vector in which words with similar meanings are close together.

</details>

**5. Why does a Transformer need positional encoding?**

<details>
<summary>Show answer</summary>

It processes all tokens in parallel, so it has no built-in sense of order. The positional encoding adds the position of each word to its embedding.

</details>

**6. Which kind of positional encoding does BERT use?**

<details>
<summary>Show answer</summary>

Learned position embeddings. The original Transformer used sinusoidal encoding; BERT and GPT use learned positions.

</details>

**7. What are the two things that are added together to form the model input?**

<details>
<summary>Show answer</summary>

The word embedding (what the word means) and the positional encoding (where the word is).

</details>

**8. What roles do the query, key, and value play?**

<details>
<summary>Show answer</summary>

The query asks "what is relevant to me?", the keys are the labels it is matched against, and the values are the content that is retrieved and mixed according to the match.

</details>

**9. Why do we divide the scores by $\sqrt{d_k}$?**

<details>
<summary>Show answer</summary>

Dot products grow with the key size. Large scores make softmax saturate and give tiny gradients. Dividing keeps the scores in a healthy range.

</details>

**10. Each row of the attention weights adds up to what?**

<details>
<summary>Show answer</summary>

1, because each row is the output of a softmax.

</details>

**11. What is the benefit of multi-head attention? What must be true about $d_{model}$ and the number of heads?**

<details>
<summary>Show answer</summary>

Different heads can learn different kinds of relationships in parallel. $d_{model}$ must be divisible by the number of heads.

</details>

**12. What is the difference between the causal mask, the padding mask, and MLM masking?**

<details>
<summary>Show answer</summary>

The causal mask hides future tokens from attention. The padding mask hides `<pad>` tokens from attention. MLM masking is not an attention mask: it replaces some input tokens with `[MASK]` so that the model must predict them.

</details>

**13. What do the residual connection, layer normalization, and feed-forward network do in a Transformer layer?**

<details>
<summary>Show answer</summary>

The residual connection adds the input back to the output (a shortcut that keeps information and helps training). Layer normalization keeps the numbers on a similar scale. The feed-forward network processes each token separately after attention has mixed information between tokens.

</details>

**14. Does a GPT-style model have encoder-decoder (cross-)attention?**

<details>
<summary>Show answer</summary>

No. Only the original encoder-decoder Transformer has cross-attention. A GPT-style model has masked self-attention only.

</details>

**15. What is self-supervised learning?**

<details>
<summary>Show answer</summary>

Learning from labels that come from the text itself (the next word, or a hidden word), so nobody has to label the data by hand.

</details>

**16. How are `tgt_input` and `tgt_out` built for teacher forcing, and why?**

<details>
<summary>Show answer</summary>

`tgt_input = tgt[:-1]` (drop the last token) and `tgt_out = tgt[1:]` (drop the first token), so each position predicts the next token. The correct previous tokens are given as input, which lets the whole sentence train in parallel.

</details>

**17. Why do we ignore the padding in the loss?**

<details>
<summary>Show answer</summary>

Padding tokens carry no information. Without `ignore_index=PAD_IDX`, the model would be trained to predict them.

</details>

**18. How does gradient accumulation let a small GPU act like a bigger one?**

<details>
<summary>Show answer</summary>

It runs several small mini-batches, adding up their gradients (several `backward()` calls), and then does one `optimizer.step()`, so the update uses a larger effective batch.

</details>

**19. What does a lower temperature do?**

<details>
<summary>Show answer</summary>

It sharpens the probabilities, so the most likely token is chosen more often (safer, more predictable text). A higher temperature flattens them (more varied text).

</details>

**20. Why does `generate` ignore `temperature` unless `do_sample=True`?**

<details>
<summary>Show answer</summary>

Without sampling, generation is greedy (always the top token), so there is nothing for temperature, top-k, or top-p to change.

</details>

**21. In MLM, why are some selected tokens replaced by a random token or left unchanged?**

<details>
<summary>Show answer</summary>

`[MASK]` never appears in the real tasks later. Randomizing 10% and leaving 10% unchanged forces the model to build a good representation of every token and reduces the mismatch between pre-training and later use.

</details>

**22. In Hugging Face's BERT next-sentence head, what does class 0 mean?**

<details>
<summary>Show answer</summary>

Sentence B is a continuation of sentence A. Class 1 means B is a random sentence.

</details>

**23. What goes wrong if `BERTEmbedding` computes `token_emb + positional_encoding(token_emb) + segment`?**

<details>
<summary>Show answer</summary>

`positional_encoding` already returns the token embedding plus the position, so the token embedding is counted twice.

</details>

**24. In the text classifier, what do `x.mean(dim=1)` and `self.classifier` do?**

<details>
<summary>Show answer</summary>

The mean pools all token vectors into one vector for the whole text. The linear layer turns that vector into one score per class (the logits).

</details>

**25. What is cross-attention, and which family has it?**

<details>
<summary>Show answer</summary>

Attention in which the queries come from the decoder's target tokens and the keys and values come from the encoder's output (the memory). Only encoder-decoder models have it.

</details>

**26. Why does BLEU have a brevity penalty?**

<details>
<summary>Show answer</summary>

Short outputs can have a very high n-gram precision (for example, a single correct word). The brevity penalty lowers the score of outputs that are shorter than the reference.

</details>

**27. Can a decoder-only model do translation?**

<details>
<summary>Show answer</summary>

Yes, large decoder-only models can translate when the task is described in the prompt. Encoder-decoder models are still a natural fit, and often smaller and cheaper, for sequence-to-sequence tasks.

</details>

### 11.6 Final words

| Family | Role |
|---|---|
| **Decoder-only (GPT)** | Generation: the foundation of many modern LLMs |
| **Encoder-only (BERT)** | Deep understanding: classification, search, tagging |
| **Encoder-decoder** | Sequence mapping: translation, summarization |

Master the shared ideas (tokens, embeddings, positions, attention, masks, and the training loop), and we can read and use any modern Transformer.

---

## Appendix: Sources and Corrections

### A. What was merged

| Source document | What it contributed | How it was used |
|---|---|---|
| **Transformers_Language_Modeling.md** | Embeddings, positional encoding, self-attention, multi-head attention, encoder and decoder, the text classifier lab (M3-L2), and training optimizations | Rebuilt in Parts 2 to 5 and Part 8, with a beginner primer added |
| **Transformer_Architectures_Manual.md** | The three families (decoder-only, encoder-only, encoder-decoder), their code, the comparison, the glossary, and the cheat sheet | Used in Parts 6, 7, 9, 10, and 11 |
| **Transformers.md** | The **original, uncorrected** version of the three-families manual | Fully covered by the corrected version above, so nothing was duplicated, and the errors it contained are **not** repeated |

Duplicates were merged: the two positional-encoding classes became one sequence-first class (Part 2.4) and one batch-first class (Part 8.3), the two glossaries and mistake tables became one each, and the self-check questions of both documents were combined.

### B. Corrections kept in this manual

| # | Where | What was wrong in an earlier version | Fix |
|---:|---|---|---|
| 1 | 1.4 | Claude and Gemini listed as decoder-only | Only models with a documented architecture are listed |
| 2 | 2.3 | "King" and "rey" said to share a point | True only for multilingual embeddings |
| 3 | 2.3 | `nn.Embedding` described as returning "pre-defined" vectors | The vectors start random and are learned |
| 4 | 2.4 | Sinusoidal encoding said to be used in BERT | BERT uses learned positions; added the three kinds |
| 5 | 3.2 | Why softmax is added to the dictionary lookup was unclear | Explained hard vs. soft lookup |
| 6 | 3.3 | Q, K, V "derived from the embedding" | They come from learned weight matrices |
| 7 | 3.5 | "`√d_k` is the dimension of the key vectors" | `d_k` is the key size; we divide by its square root |
| 8 | 3.8 | Multi-head: "the embedding is split into h parts" | `d_k = d_model / h`; it must divide evenly |
| 9 | 3.9 | MLM masking listed as an attention mask | It is data corruption, not an attention mask |
| 10 | 4.3 | GPT-like decoder said to have encoder-decoder attention | GPT-style models have no cross-attention |
| 11 | 5.2 | Cross-entropy "for next-token prediction (MLM, LM)" | Separated the three tasks |
| 12 | 5.5 | `torch.cuda.amp` API | `torch.autocast` and `torch.amp.GradScaler` |
| 13 | 6.3 | `create_mask(x)` called with embeddings | The causal mask is built from the sequence length |
| 14 | 6.8 | `temperature`, `top_k`, `top_p` without `do_sample=True`; `max_length` counts the prompt | Added `do_sample=True` and `max_new_tokens` |
| 15 | 7.1 | RoBERTa shown with MLM + NSP | Later models dropped NSP |
| 16 | 7.3 | The 80/10/10 explanation; 20% in code vs. 15% in the text | Explained the mismatch problem; noted both rates |
| 17 | 7.4 | `process_for_nsp` never filled `bert_label` | Filled the labels |
| 18 | 7.5 | `BERTEmbedding` counted the token embedding twice | Removed the extra term |
| 19 | 7.9 | NSP class 0 read as "NOT consecutive"; `argmax` without `dim` | In Hugging Face, 0 = continuation; added `dim=-1` |
| 20 | 8.3 | The positional-encoding argument `vocab_size` is really the maximum length | Renamed `max_len`; noted the even `d_model` |
| 21 | 8.4 | `batch_first` not stated; the mean pooling also averages padding | Added `batch_first=True` and a padding-mask note |
| 22 | 8.5 | Comment says AdamW but the code uses SGD; `scheduler.step()` never called | Flagged both |
| 23 | 9.5 | "Three masks" but four tensors returned | Three kinds of mask, four tensors |
| 24 | 9.7 | `len(list(train_dataloader))` loaded the data a second time | `len(train_dataloader)` |
| 25 | 9.10 | BLEU without smoothing or consistent case and tokenization | Added lowercasing and smoothing |
| 26 | 10.2 | The decision flowchart ignored that decoder-only LLMs also translate and classify | Added a note |
| 27 | 11.3 | The generation snippet could not concatenate a 0-dimensional tensor | Used `keepdim=True` |
| 28 | 11.3 | The decoder-only attention matrix was drawn as an upper triangle | Redrawn as a lower triangle |

### C. Which lab covers what

| Lab (named by its topic) | Parts |
|---|---|
| **Attention Mechanism and Positional Encoding** | 2.4, 3.2 to 3.9 |
| **M3-L2: Applying Transformers for Classification** | 2.3, 4.1, 5.5, 8 |
| **GPT-like model lab** (decoder-only) | 6 |
| **BERT-like model lab** (encoder-only) | 7 |
| **Translation lab** (encoder-decoder) | 9 |
