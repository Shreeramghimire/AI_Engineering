# Neural Networks for Text: A Simple Learning Manual
### From words and numbers, to language models, Word2Vec, and translation

> **How to use this manual:** read the parts in order, because each part builds on the one before it. Every part starts with a short idea, uses small examples and tables, and ends with a summary. Open the **Self-check** in Part 9 to test yourself. The code is short on purpose, and each lab is named at the start of the part that uses it.

## Table of Contents

- **[Learning Path](#learning-path)**
  - [The big picture](#the-big-picture)
- **[Part 1: Turning Words into Numbers](#part-1-turning-words-into-numbers)**
  - [1.1 Why do we need numbers?](#11-why-do-we-need-numbers)
  - [1.2 Tokens and a vocabulary](#12-tokens-and-a-vocabulary)
  - [1.3 One-hot encoding](#13-one-hot-encoding)
  - [1.4 Bag of words](#14-bag-of-words)
  - [1.5 Embeddings](#15-embeddings)
  - [1.6 Embeddings in PyTorch](#16-embeddings-in-pytorch)
  - [1.7 Part 1 summary](#17-part-1-summary)
- **[Part 2: Neural Network Basics for Text](#part-2-neural-network-basics-for-text)**
  - [2.1 What is a neural network?](#21-what-is-a-neural-network)
  - [2.2 Learnable parameters and hyperparameters](#22-learnable-parameters-and-hyperparameters)
  - [2.3 Logits, probabilities, and argmax](#23-logits-probabilities-and-argmax)
  - [2.4 Loss and cross-entropy](#24-loss-and-cross-entropy)
  - [2.5 Optimization](#25-optimization)
  - [2.6 Monte Carlo sampling](#26-monte-carlo-sampling)
  - [2.7 Part 2 summary](#27-part-2-summary)
- **[Part 3: Text Classification](#part-3-text-classification)**
  - [3.1 What is document classification?](#31-what-is-document-classification)
  - [3.2 The pipeline](#32-the-pipeline)
  - [3.3 Managing the data](#33-managing-the-data)
  - [3.4 The model and weight initialization](#34-the-model-and-weight-initialization)
  - [3.5 The training loop](#35-the-training-loop)
  - [3.6 The prediction function](#36-the-prediction-function)
  - [3.7 Part 3 summary](#37-part-3-summary)
- **[Part 4: Language Models and N-grams](#part-4-language-models-and-n-grams)**
  - [4.1 What is a language model?](#41-what-is-a-language-model)
  - [4.2 Counting models: unigram, bigram, trigram](#42-counting-models-unigram-bigram-trigram)
  - [4.3 Why counting models struggle](#43-why-counting-models-struggle)
  - [4.4 Neural n-gram models](#44-neural-n-gram-models)
  - [4.5 Training and generating](#45-training-and-generating)
  - [4.6 Which score should we watch?](#46-which-score-should-we-watch)
  - [4.7 Part 4 summary](#47-part-4-summary)
- **[Part 5: Word2Vec](#part-5-word2vec)**
  - [5.1 What is Word2Vec?](#51-what-is-word2vec)
  - [5.2 The architecture](#52-the-architecture)
  - [5.3 CBOW (Continuous Bag of Words)](#53-cbow-continuous-bag-of-words)
  - [5.4 Skip-gram](#54-skip-gram)
  - [5.5 CBOW or Skip-gram?](#55-cbow-or-skip-gram)
  - [5.6 Training Word2Vec and getting the embeddings](#56-training-word2vec-and-getting-the-embeddings)
  - [5.7 Looking at the embeddings](#57-looking-at-the-embeddings)
  - [5.8 Pretrained embeddings: GloVe and gensim](#58-pretrained-embeddings-glove-and-gensim)
  - [5.9 Using pretrained embeddings for classification](#59-using-pretrained-embeddings-for-classification)
  - [5.10 Part 5 summary](#510-part-5-summary)
- **[Part 6: Sequence Models and RNNs](#part-6-sequence-models-and-rnns)**
  - [6.1 Three kinds of sequence tasks](#61-three-kinds-of-sequence-tasks)
  - [6.2 Why do we need memory?](#62-why-do-we-need-memory)
  - [6.3 What is an RNN?](#63-what-is-an-rnn)
  - [6.4 Limits of RNNs, and LSTM and GRU](#64-limits-of-rnns-and-lstm-and-gru)
  - [6.5 Part 6 summary](#65-part-6-summary)
- **[Part 7: Encoder-Decoder Models for Translation](#part-7-encoder-decoder-models-for-translation)**
  - [7.1 The idea](#71-the-idea)
  - [7.2 The encoder](#72-the-encoder)
  - [7.3 The decoder](#73-the-decoder)
  - [7.4 Connecting them: the Seq2Seq model and teacher forcing](#74-connecting-them-the-seq2seq-model-and-teacher-forcing)
  - [7.5 The data](#75-the-data)
  - [7.6 Training](#76-training)
  - [7.7 Translating a new sentence](#77-translating-a-new-sentence)
  - [7.8 Why are sequence-to-sequence models harder to train?](#78-why-are-sequence-to-sequence-models-harder-to-train)
  - [7.9 Part 7 summary](#79-part-7-summary)
- **[Part 8: Evaluating Language and Translation Models](#part-8-evaluating-language-and-translation-models)**
  - [8.1 Which score for which job?](#81-which-score-for-which-job)
  - [8.2 Perplexity](#82-perplexity)
  - [8.3 Precision, recall, and F1 for translation](#83-precision-recall-and-f1-for-translation)
  - [8.4 BLEU score](#84-bleu-score)
  - [8.5 Libraries](#85-libraries)
  - [8.6 Part 8 summary](#86-part-8-summary)
- **[Part 9: Putting It All Together](#part-9-putting-it-all-together)**
  - [9.1 The big map](#91-the-big-map)
  - [9.2 The five labs at a glance](#92-the-five-labs-at-a-glance)
  - [9.3 Watch-outs found in the labs](#93-watch-outs-found-in-the-labs)
  - [9.4 Glossary](#94-glossary)
  - [9.5 Self-check](#95-self-check)

---

## Learning Path

| Part | Topic | The big question | Lab |
|---|---|---|---|
| **1** | Turning words into numbers | How can a computer read words? | |
| **2** | Neural network basics for text | How does a network learn? | |
| **3** | Text classification | How do we sort articles into topics? | Feed Forward Neural Networks |
| **4** | Language models and N-grams | How do we predict the next word? | Language Modelling, Feed Forward Neural Networks |
| **5** | Word2Vec | How do we learn good word vectors? | Integrating Word2Vec, Part 1 and Part 2 |
| **6** | Sequence models and RNNs | How do we handle order and memory? | Sequence-to-Sequence Model |
| **7** | Encoder-decoder models | How do we translate one sentence into another? | Sequence-to-Sequence Model |
| **8** | Evaluating the models | How do we know if the model is good? | Feed Forward Neural Networks, Sequence-to-Sequence Model |
| **9** | Putting it all together | What should we remember? | All five labs |

### The big picture

```text
Words → tokens → numbers (indexes) → embeddings (vectors) → neural network → scores (logits) → prediction

Classification:      article        →  one topic
Language model:      some words     →  the next word
Word2Vec:            a word         →  its neighbors (and we keep the embeddings)
Translation:         a sentence     →  another sentence (encoder-decoder)
```

Every part of this manual uses the same simple recipe: **turn text into numbers, let a network make a guess, measure the mistake (loss), and adjust the network to make the mistake smaller.**

---

## Part 1: Turning Words into Numbers

### 1.1 Why do we need numbers?

A computer cannot read words. It can only do math on numbers. So before a neural network can work with text, we must change every word into numbers. This part shows the steps, from the simplest idea (a list of zeros and ones) to the best idea (embeddings).

### 1.2 Tokens and a vocabulary

1. **Tokenization:** cut the text into small pieces called **tokens** (usually words).
2. **Vocabulary:** make a list of all different tokens. Give each token its own number (its **index**).
3. **Special tokens:** we add a few extra tokens with special jobs.

| Special token | Meaning | Why we need it |
|---|---|---|
| `<unk>` | Unknown word | For words that are not in our vocabulary |
| `<pad>` | Padding | To make short sentences as long as the longest one in a batch |
| `<bos>` | Beginning of sentence | Tells a model where a sentence starts |
| `<eos>` | End of sentence | Tells a model where a sentence ends |

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("The cat sat on the mat")        # ['the', 'cat', 'sat', 'on', 'the', 'mat']

vocab = build_vocab_from_iterator([tokens], specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])               # unknown words become <unk>

print(vocab(tokens))                                  # the words as numbers
print(vocab.get_itos()[1])                            # a number back to its word
```

> **Note:** the labs use `torchtext`, which is no longer actively developed, so we keep the library versions that the labs use (PyTorch 2.2.2 and torchtext 0.17.2).

### 1.3 One-hot encoding

**One-hot encoding** turns each word into a long list of zeros with a single one. The position of the one tells us which word it is.

Suppose our vocabulary has three words: `cat`, `dog`, `fish`.

| Word | One-hot vector |
|---|---|
| cat | `[1, 0, 0]` |
| dog | `[0, 1, 0]` |
| fish | `[0, 0, 1]` |

This is simple, but it has two problems:

- The vectors are as long as the vocabulary (thousands of words means thousands of numbers).
- All words look equally different. The vectors for `cat` and `kitten` are no closer than `cat` and `car`.

### 1.4 Bag of words

A **bag of words** describes a whole document by adding up (or averaging) the one-hot vectors of its words. We forget the order of the words, as if the words were thrown into a bag.

For the document **"dog dog cat"**:

```text
cat  = [1, 0, 0]
dog  = [0, 1, 0]
dog  = [0, 1, 0]
------------------
sum  = [1, 2, 0]      ← the bag of words (cat once, dog twice, fish never)
```

### 1.5 Embeddings

An **embedding** gives every word a short list of numbers (for example 50 or 300 numbers) instead of a long one-hot vector. These numbers are **learned** during training, so the network can place similar words close together.

| | One-hot | Embedding |
|---|---|---|
| **Length** | As long as the vocabulary | Short (we choose it, for example 20) |
| **Values** | Only 0 and 1 | Any decimal numbers |
| **Similar words** | Not close | Can be close |
| **Learned?** | No | Yes |

Here is the key idea from the course: **if we feed a bag-of-words vector into a neural network's first layer, the result is the sum of the embeddings of the words** (an embedding layer does the same job, but faster, because it just looks up the rows we need).

A tiny example with 2-number embeddings:

| Word | Embedding |
|---|---|
| cat | `[0.2, 0.8]` |
| dog | `[0.6, -0.5]` |
| fish | `[-0.3, 0.7]` |

Document "dog dog cat": sum = `[0.6, -0.5] + [0.6, -0.5] + [0.2, 0.8] = [1.4, -0.2]`. (If we use the average instead of the sum, we divide by 3.)

At the start, the embeddings are random. As the model trains, words that are used in similar ways get similar vectors.

### 1.6 Embeddings in PyTorch

| Class | What it does | Output |
|---|---|---|
| `nn.Embedding` | Looks up one vector for **each word** | One vector per word |
| `nn.EmbeddingBag` | Looks up the vectors and **combines them** (average by default, or sum) | One vector per **document** |

```python
import torch
import torch.nn as nn

emb = nn.Embedding(num_embeddings=5, embedding_dim=3)       # 5 words, 3 numbers each
print(emb(torch.tensor([1, 4])).shape)                       # torch.Size([2, 3]): one vector per word

bag = nn.EmbeddingBag(5, 3, mode="mean")
words   = torch.tensor([1, 4, 2, 0, 3])                      # two documents written one after the other
offsets = torch.tensor([0, 2])                               # document 1 starts at 0, document 2 starts at 2
print(bag(words, offsets).shape)                             # torch.Size([2, 3]): one vector per document
```

The **offsets** tell `EmbeddingBag` where each document starts, so we can put many documents of different lengths into one long list.

### 1.7 Part 1 summary

| Idea | One-sentence summary |
|---|---|
| **Token and vocabulary** | Cut text into pieces and give each piece a number |
| **One-hot encoding** | A long vector with a single 1 |
| **Bag of words** | The sum or average of one-hot vectors; the order is lost |
| **Embedding** | A short, learned vector for each word |
| **`Embedding` / `EmbeddingBag`** | One vector per word / one vector per document |

---

## Part 2: Neural Network Basics for Text

### 2.1 What is a neural network?

A **neural network** is a math function. It is built from a chain of **matrix multiplications** with other simple functions in between (such as ReLU). We give it numbers, it gives us numbers back.

```text
numbers in → multiply by matrix → simple function → multiply by matrix → ... → numbers out
```

### 2.2 Learnable parameters and hyperparameters

| | **Learnable parameters** | **Hyperparameters** |
|---|---|---|
| **What they are** | The numbers inside the matrices and vectors (weights and biases, and the embeddings) | Settings we choose **before** training |
| **Who sets them** | The training process, automatically | We do |
| **Examples** | Weights, biases, embedding vectors | Learning rate, batch size, number of epochs, embedding size, context size |
| **When they change** | Many times during training | Only when we run a new experiment |

During training the learnable parameters are fine-tuned to make the model better. The **loss function** guides this: it is a score that tells us how wrong the model is.

### 2.3 Logits, probabilities, and argmax

For classification, the last layer gives one number per class. These raw numbers are called **logits**. They can be any size, positive or negative.

| Step | What happens | Example |
|---|---|---|
| **Logits** | Raw scores from the network | `[2.0, 1.0, 0.1]` |
| **Softmax** (optional) | Turns the scores into probabilities that add up to 1 | `[0.66, 0.24, 0.10]` |
| **Argmax** | Picks the **index** of the largest value | `0` (the first class) |

The **argmax** function returns the position of the biggest number, which is the most likely class. We can take the argmax of the logits directly, because the largest logit also gives the largest probability.

```python
logits = torch.tensor([[2.0, 1.0, 0.1]])
predicted_class = torch.argmax(logits, dim=1)       # tensor([0])
```

### 2.4 Loss and cross-entropy

The **loss** is one number that says how wrong the model is. For choosing among classes (or words), we use **cross-entropy**. It looks at the probability the model gave to the **correct** answer:

| Probability given to the correct answer | Loss | Meaning |
|---:|---:|---|
| 0.9 | 0.105 | The model was confident and right: small loss |
| 0.5 | 0.693 | The model was unsure: medium loss |
| 0.1 | 2.303 | The model was confident and wrong: big loss |

In PyTorch, `nn.CrossEntropyLoss` already includes the softmax step, so **we give it the raw logits**, not probabilities.

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target_class_indexes)       # logits in, class indexes as targets
```

**Why cross-entropy?** Finding the best parameters means finding the ones that give the **correct answers a high probability**. Cross-entropy measures exactly that, so making it small makes the model better.

### 2.5 Optimization

**Optimization** means making the loss smaller. We do it in small steps:

1. Make a prediction.
2. Measure the loss.
3. Find out which way each parameter should move to make the loss smaller (the **gradient**).
4. Move every parameter a small step in that direction.
5. Repeat many times.

The size of each step is the **learning rate**. A step that is too big overshoots; a step that is too small is slow. In PyTorch we use an **optimizer** (such as SGD or Adam) to do step 4.

### 2.6 Monte Carlo sampling

Sometimes we want the average of something over **all** possible cases, but we cannot look at all of them. **Monte Carlo sampling** is the simple fix: **take a set of samples, apply the function to each, and average the results.**

> **Example:** we want the average height of all people in a country. We cannot measure everyone, so we measure 1,000 random people and average. The more samples, the better the estimate.

This is exactly what happens in training. The true loss is an average over **all possible text**, which is unknown. We estimate it by averaging the loss over the **training samples** in each batch. That is why a bigger, more varied dataset gives a better estimate.

### 2.7 Part 2 summary

| Idea | One-sentence summary |
|---|---|
| **Neural network** | A chain of matrix multiplications and simple functions |
| **Learnable parameters** | Numbers the training process changes |
| **Hyperparameters** | Settings we choose before training |
| **Logits** | Raw output scores |
| **Argmax** | Index of the largest score = the predicted class |
| **Cross-entropy** | Loss that gives a high penalty when the correct answer gets a low probability |
| **Optimization** | Small steps that make the loss smaller |
| **Monte Carlo sampling** | Estimate an average by averaging over samples |

---

## Part 3: Text Classification

> **Lab for this part:** Feed Forward Neural Networks (the data steps, the training loop, and the model style). The classification example below follows the AG News example from the Word2Vec lab, Part 2.

### 3.1 What is document classification?

A **document classifier** reads the text of an article and puts it into a category. For example, **AG News** has four categories:

| Label | Category |
|---:|---|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tec |

### 3.2 The pipeline

```text
raw text → lowercase and tokenize → indexes → EmbeddingBag (average the word vectors)
        → Linear layer → logits (one score per category) → argmax → predicted category
```

The model is small: an **embedding bag** turns the document into one vector, and a **linear layer** turns that vector into one score for each category.

### 3.3 Managing the data

#### Three subsets

| Subset | Used for | Rule |
|---|---|---|
| **Training data** | Learning (the weights change) | The model sees it again and again |
| **Validation data** | Tuning hyperparameters and checking progress | The weights do **not** learn from it |
| **Test data** | A final, honest score of real-world performance | Use it once, at the end |

We first split the training data into a training part and a validation part (for example 85% and 15%). The test data stays separate.

#### Data loaders and batching

A **data loader** hands the data to the model in small groups called **batches**.

- **Batch size** is the number of samples used to estimate the gradient for one update. Small batches give a noisier but quicker signal; large batches are smoother but use more memory.
- **Shuffling** the training data in every epoch helps the optimizer, because the model does not see the samples in the same order each time. We do not need to shuffle validation or test data.

Documents have different lengths. So a **collate function** joins all the documents in a batch into one long list and records where each document starts (the **offsets**), which is exactly what `EmbeddingBag` expects.

```python
def collate_batch(batch):
    labels, texts, offsets = [], [], [0]
    for label, text in batch:
        labels.append(int(label) - 1)                          # AG News labels are 1 to 4, PyTorch wants 0 to 3
        ids = torch.tensor(text_pipeline(text), dtype=torch.int64)
        texts.append(ids)
        offsets.append(ids.size(0))
    labels  = torch.tensor(labels, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)         # start position of each document
    texts   = torch.cat(texts)
    return labels, texts, offsets
```

```python
from torch.utils.data import DataLoader, random_split

num_train = int(len(train_dataset) * 0.85)
split_train, split_valid = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_loader = DataLoader(split_train, batch_size=64, shuffle=True,  collate_fn=collate_batch)
valid_loader = DataLoader(split_valid, batch_size=64, shuffle=False, collate_fn=collate_batch)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)
```

### 3.4 The model and weight initialization

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)      # document → one vector
        self.fc = nn.Linear(embed_dim, num_classes)                  # vector → one score per class
        self.init_weights()

    def init_weights(self):                                          # starting values for the weights
        self.embedding.weight.data.uniform_(-0.5, 0.5)
        self.fc.weight.data.uniform_(-0.5, 0.5)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        return self.fc(self.embedding(text, offsets))                # logits
```

`init_weights` gives the model good starting values: small random numbers for the weights, and zeros for the bias. A good start helps the optimization.

### 3.5 The training loop

The course lists the steps of the training loop. Here they are in the order we actually run them:

| Step | What we do |
|---:|---|
| 1 | Go through the data **epoch by epoch** (one epoch = one full pass over the training data) |
| 2 | Put the model in **training mode** (`model.train()`) |
| 3 | Take the data **batch by batch** |
| 4 | **Predict** and calculate the **loss** for the batch |
| 5 | Do **gradient descent**: clear old gradients, compute new ones, update the weights |
| 6 | **Add up the loss** after each batch, and take the average at the end of the epoch |
| 7 | Check the **validation data** (evaluation mode, no learning) to see the progress |

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for labels, texts, offsets in train_loader:
        optimizer.zero_grad()                                        # clear old gradients
        loss = criterion(model(texts, offsets), labels)              # predict, then measure the mistake
        loss.backward()                                              # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)      # keep the gradients from getting too large
        optimizer.step()                                             # update the weights
        total_loss += loss.item()
    print(f"epoch {epoch}: average loss = {total_loss / len(train_loader):.4f}")
```

```python
def evaluate(loader):
    model.eval()                                                     # evaluation mode
    correct, count = 0, 0
    with torch.no_grad():                                            # no learning here
        for labels, texts, offsets in loader:
            predicted = model(texts, offsets).argmax(1)
            correct += (predicted == labels).sum().item()
            count += labels.size(0)
    return correct / count
```

### 3.6 The prediction function

A **prediction function** works on real text. It takes the raw text, tokenizes it, runs it through the same pipeline and model, and returns the category.

```python
ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}

def predict(text, text_pipeline, model):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor(text_pipeline(text), dtype=torch.int64)
        logits = model(ids, torch.tensor([0]))                       # one document, so one offset
        return ag_news_label[logits.argmax(1).item() + 1]            # argmax gives 0..3, labels are 1..4
```

### 3.7 Part 3 summary

| Idea | One-sentence summary |
|---|---|
| **Document classifier** | Sorts articles into categories by reading their text |
| **Train / validation / test** | Learn / tune / final honest score |
| **Data loader and batching** | Feed the data in small groups; shuffle the training data |
| **`init_weights`** | Good starting values help optimization |
| **Training loop** | Epoch → batch → predict → loss → gradient descent → average the loss |
| **Prediction function** | Raw text → pipeline → model → argmax → category name |

---

## Part 4: Language Models and N-grams

> **Labs for this part:** Language Modelling (counting models) and Feed Forward Neural Networks (neural models).

### 4.1 What is a language model?

A **language model** says how likely a word, or a sequence of words, is. The most useful version answers: **"Given the words so far, which word comes next?"** It is like the autocomplete on a phone. Language models are used for text generation, translation, and speech recognition.

### 4.2 Counting models: unigram, bigram, trigram

The simplest language models just **count** words in a text. We use the lab's tiny example text:

> **"I like dogs and I kinda like cats"** (8 words)

#### Unigram model (no context)

It looks at each word alone: $P(\text{word}) = \frac{\text{Count(word)}}{\text{Total words}}$

For example, $P(\text{"I"}) = \frac{2}{8} = 0.25$. If we ask for the next word after "I like", a unigram model just returns the most common word ("I"), which makes no sense. **It has no context.**

#### Bigram model (context size 1)

A **bigram** is a pair of neighboring words. The model uses only the **one previous word**:

$$P(w_t \mid w_{t-1}) = \frac{\text{Count}(w_{t-1}, w_t)}{\text{Count}(w_{t-1})}$$

The bigrams of our text are: (I, like), (like, dogs), (dogs, and), (and, I), (I, kinda), (kinda, like), (like, cats).

| Question | Calculation | Answer |
|---|---|---|
| $P(\text{like} \mid \text{I})$ | Count(I, like) / Count(I) = 1 / 2 | 0.5 |
| $P(\text{dogs} \mid \text{like})$ | 1 / 2 | 0.5 |
| $P(\text{cats} \mid \text{like})$ | 1 / 2 | 0.5 |
| $P(\text{like} \mid \text{kinda})$ | 1 / 1 | 1.0 |

After "like", "dogs" and "cats" are equally likely, so the model cannot choose well. To predict, we take the word with the highest probability (**argmax**).

#### Trigram and N-gram models

- A **trigram** model uses the **two** previous words (context size 2): $P(w_t \mid w_{t-2}, w_{t-1})$. More context improves on the bigram's limits.
- An **N-gram** model uses an **arbitrary** context size (N − 1 previous words).

| Model | Context size | It looks at |
|---|---:|---|
| Unigram | 0 | Nothing before the word |
| Bigram | 1 | The previous word |
| Trigram | 2 | The previous two words |
| N-gram | N − 1 | The previous N − 1 words |

```python
import nltk

tokens = ["i", "like", "dogs", "and", "i", "kinda", "like", "cats"]
word_counts   = nltk.FreqDist(tokens)
bigram_counts = nltk.FreqDist(nltk.bigrams(tokens))

p = bigram_counts[("i", "like")] / word_counts["i"]       # P(like | i) = 0.5
```

To **generate text**, we pick the most likely next word, add it to the text, and repeat with the new word as the context.

### 4.3 Why counting models struggle

| Problem | What it means |
|---|---|
| **Too many counts** | With N words in the vocabulary, a unigram model needs N counts, a bigram N², a trigram N³, and so on |
| **Unseen word groups** | A word pair that never appeared in the text gets probability 0 |
| **No meaning** | The model does not know that "don't like" means "dislike" |
| **Repetition** | In the lab, the bigram model got stuck repeating the same phrase over and over |

### 4.4 Neural n-gram models

A **neural n-gram model** fixes these problems with **embeddings**. It is simply a **classification model** where the classes are the words in the vocabulary.

```text
previous words → embeddings → join them side by side (concatenate) → hidden layer (ReLU) → output layer → one score for every word in the vocabulary
```

**Input size.** With one-hot vectors, the input would have (vocabulary size × context size) numbers, which is huge. In practice we avoid this: we look up an embedding for each context word and join them side by side. So the input size is only **context size × embedding size**.

**Example from the lab:** vocabulary of 79 words, embedding size 10, context size 3:

| Layer | Size | Parameters |
|---|---|---:|
| Embeddings | 79 words × 10 numbers | 790 |
| Hidden layer | 30 inputs (3 × 10) → 128 | 3,968 |
| Output layer | 128 → 79 scores | 10,191 |
| **Total** | | **14,949** |

A bigger context only changes the hidden layer: context 2 gives 13,669 parameters, context 4 gives 16,229, and context 8 gives 21,349.

```python
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.context_size, self.embedding_dim = context_size, embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)       # hidden layer
        self.linear2 = nn.Linear(128, vocab_size)                         # one score per word

    def forward(self, inputs):                                            # inputs: (batch, context_size) word indexes
        embeds = self.embeddings(inputs)                                  # (batch, context_size, embedding_dim)
        embeds = embeds.reshape(-1, self.context_size * self.embedding_dim)   # join the context embeddings
        out = torch.relu(self.linear1(embeds))
        return self.linear2(out)                                          # logits over the vocabulary
```

When the lab fed 7 context-target samples into an untrained model, the output had shape `[7, 79]`: one score for each of the 79 words, for each sample.

#### The sliding window

To make training data, we **slide a window** along the text, one word at a time. Each position gives one **(context, target)** pair. For "the cat sat on the mat" with context size 2:

| Context | Target |
|---|---|
| the, cat | sat |
| cat, sat | on |
| sat, on | the |
| on, the | mat |

### 4.5 Training and generating

| Setting in the lab | Value |
|---|---|
| Loss | Cross-entropy |
| Optimizer | SGD, learning rate 0.01, with a step learning-rate scheduler |
| Batch size | 10 |
| Epochs | 100 |
| Embedding size | 10 |
| Context sizes tried | 2, 4, and 8 |

To **generate text** after training: start with a few seed words, predict the next word (argmax), add it, move the context window by one word, and repeat.

```python
def next_word(model, vocab, last_words):
    # last_words: the most recent words, in the same order as the training contexts
    ids = torch.tensor([vocab(last_words)])
    with torch.no_grad():
        logits = model(ids)
    return vocab.get_itos()[logits.argmax(1).item()]
```

**What the lab showed:**

- Early in training, the generated text was mostly the `<unk>` token. Later it started to look like the training text.
- A **larger context gave a lower training loss** (the 8-gram model was lowest).

> **Be careful:** the lab text is tiny (79 different words) and has no validation set. A lower loss with a bigger context mostly means the model **memorizes** the text, not that it understands language. For real work, check the loss on validation data.

### 4.6 Which score should we watch?

When training language models, the course says to prioritize the **loss** over **accuracy** as the key performance indicator (KPI).

| | **Accuracy** | **Loss** |
|---|---|---|
| What it counts | Only whether the top guess is exactly right | How much probability went to the right word |
| Problem | Many different next words can be fine, and a near miss counts as a complete miss | Smooth: it improves even when the top guess is still wrong |
| Use | Easy to explain | Better to guide and judge training |

We turn the loss into **perplexity** in Part 8.

### 4.7 Part 4 summary

| Idea | One-sentence summary |
|---|---|
| **Language model** | Predicts the next word from the words before it |
| **Unigram / bigram / trigram** | Context size 0 / 1 / 2 |
| **N-gram** | Any context size |
| **Counting models** | Easy, but huge, blind to meaning, and they repeat themselves |
| **Neural n-gram model** | Embeddings of the context words → hidden layer → one score per word |
| **Sliding window** | Move along the text one word at a time to build (context, target) pairs |
| **KPI** | Watch the loss, not accuracy |

---

## Part 5: Word2Vec

> **Labs for this part:** Integrating Word2Vec, Part 1 (CBOW and Skip-gram) and Part 2 (pretrained embeddings and gensim).

### 5.1 What is Word2Vec?

**Word2Vec** means "word to vector". It is a group of models that learn **word embeddings**: lists of numbers that capture the essence of a word. The main idea is simple:

> **Words that appear in similar places have similar meanings, so they get similar vectors.**

For example, the vectors for "cat", "kitten", and "dog" end up close together, while the vector for "book" is far away. Once words are vectors, we can measure how similar two words are with math.

### 5.2 The architecture

A Word2Vec network has three layers:

| Layer | Size | Job |
|---|---|---|
| **Input layer** | Vocabulary size | Receives the word |
| **Embedding layer** | **We choose it** (for example 24, 100, or 300): this is the length of the word vector | Holds the word vectors |
| **Output layer** | Vocabulary size | Gives a score for every word in the vocabulary |

The word goes into the embedding layer, which talks to the output layer to **predict the context words** (the neighbors). We do not care about the predictions themselves. **After training, we keep the weights of the embedding layer. Those weights are the word embeddings.**

There are two ways to set up the prediction, and they are opposites: **CBOW** and **Skip-gram**.

### 5.3 CBOW (Continuous Bag of Words)

**CBOW uses the context words to predict the target word.** We use a **window**: for a window size of 2, we take two words before and two words after the target.

Take the sentence **"I wish I was little bit taller"** with window size 2:

| Context (4 words) | Target |
|---|---|
| i, wish, was, little | i |
| wish, i, little, bit | was |
| i, was, bit, taller | little |

We slide the window along the text, one word at a time, to create all the training samples.

```python
CONTEXT_SIZE = 2
cbow_data = []
for i in range(CONTEXT_SIZE, len(tokens) - CONTEXT_SIZE):
    context = ([tokens[i - CONTEXT_SIZE + j] for j in range(CONTEXT_SIZE)]      # words before
               + [tokens[i + j + 1] for j in range(CONTEXT_SIZE)])              # words after
    target = tokens[i]
    cbow_data.append((context, target))
```

The model averages the embeddings of the context words (with `EmbeddingBag`), then uses two linear layers to produce a score for every word in the vocabulary:

```python
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)       # averages the context word vectors
        self.linear1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc = nn.Linear(embed_dim // 2, vocab_size)               # one score per word

    def forward(self, text, offsets):
        out = self.embedding(text, offsets)
        out = torch.relu(self.linear1(out))
        return self.fc(out)
```

### 5.4 Skip-gram

**Skip-gram does the opposite: it uses the target word to predict the surrounding context words.**

Predicting all the neighbors at once is expensive. So Skip-gram makes it simpler: **it predicts one context word at a time** from the target word. The target "little" in our sentence gives four separate training pairs:

| Target word (input) | Context word (what to predict) |
|---|---|
| little | was |
| little | i |
| little | bit |
| little | taller |

The code that builds the samples is almost the same as for CBOW. We just **switch the roles of target and context** and split the full context into smaller parts, one pair per context word.

```python
skip_data = []
for i in range(CONTEXT_SIZE, len(tokens) - CONTEXT_SIZE):
    context = ([tokens[i - j - 1] for j in range(CONTEXT_SIZE)]                 # words before
               + [tokens[i + j + 1] for j in range(CONTEXT_SIZE)])              # words after
    skip_data += [(tokens[i], word) for word in context]                        # (target, one context word)
```

```python
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)         # the word vectors we want
        self.fc = nn.Linear(embed_dim, vocab_size)                    # one score per word

    def forward(self, target):
        return self.fc(self.embeddings(target))
```

> **Note:** the lab version also applies a ReLU to the embedding before the last layer. The idea is the same.

### 5.5 CBOW or Skip-gram?

| | **CBOW** | **Skip-gram** |
|---|---|---|
| **Input** | All the context words | One target word |
| **Predicts** | The target word | One context word at a time |
| **Training samples** | Fewer (one per position) | More (one per context word) |
| **Strength** | Faster and cheaper | Better for **rare words** and small datasets |
| **Choose it when** | Speed matters and rare words matter less | The dataset is small but we need good vectors for rarer words |

For **very small** datasets, neither gives great vectors. Then, simpler methods or **pretrained embeddings** may work better (Section 5.8).

### 5.6 Training Word2Vec and getting the embeddings

When we train, we choose four things:

| Choice | In the lab |
|---|---|
| **Learning rate** | 5 (high, because the data is a tiny toy example) |
| **Loss function** | Cross-entropy |
| **Optimizer** | SGD |
| **Learning rate scheduler** | A step scheduler that lowers the learning rate |

The lab trains for 400 epochs with batch size 64 and an embedding size of 24.

When training is done, we take the weights. **They are the embeddings.**

```python
word_embeddings = model_cbow.embedding.weight.detach().cpu().numpy()      # a table: one row per word
vector = word_embeddings[vocab.get_stoi()["baller"]]                       # 24 numbers for this word
```

### 5.7 Looking at the embeddings

#### t-SNE: a 2D picture

An embedding has many numbers, so we cannot draw it. **t-SNE** squeezes the vectors down to 2D so that we can plot them. In the lab's plot, even with a tiny dataset, words with similar meaning landed near each other (for example "dog", "cat", and "mouse").

#### Cosine similarity: a number for "how close"

**Cosine similarity** compares the *direction* of two vectors. It is 1 for the same direction, 0 for unrelated, and −1 for opposite.

$$\text{similarity}(a, b) = \frac{a \cdot b}{\|a\| \, \|b\|}$$

```python
import numpy as np

def find_similar_words(target, embedding_dict, top_k=2):
    t = embedding_dict[target]
    scores = {w: np.dot(t, v) / (np.linalg.norm(t) * np.linalg.norm(v))
              for w, v in embedding_dict.items() if w != target}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

### 5.8 Pretrained embeddings: GloVe and gensim

Training good embeddings needs a **lot** of text. Instead, we can reuse vectors that someone else trained on huge amounts of text. This is **transfer learning**: knowledge learned on one big task is reused on our smaller task. It saves computing time and gives the model a head start in understanding language.

#### GloVe

**GloVe** (Global Vectors) is a popular set of pretrained embeddings from Stanford. Word2Vec learns by **predicting neighbors**; GloVe learns from **how often words appear together** in the whole text (a co-occurrence table). If two words often appear together, their vectors are similar.

```python
from torchtext.vocab import GloVe

glove = GloVe(name="6B", dim=300)                                           # a large download
embeddings = torch.nn.Embedding.from_pretrained(glove.vectors, freeze=True) # freeze = do not change them
index = glove.stoi["team"]                                                  # the row for the word "team"
vector = embeddings.weight[index]
```

| GloVe model | Size |
|---|---|
| `6B` | Smaller and lighter: the lab continues with this one |
| `840B` | Much bigger (300 numbers per word) |

#### gensim

The `gensim` library trains Word2Vec for us in a few lines:

```python
from gensim.models import Word2Vec

sentences = [["i", "like", "to", "eat", "pizza"],
             ["pizza", "is", "my", "favorite", "food"],
             ["i", "enjoy", "eating", "pasta"]]

w2v = Word2Vec(sentences, vector_size=100, window=3, min_count=1, workers=4, epochs=30)
print(w2v.wv.most_similar("pizza"))
```

| Parameter | Meaning |
|---|---|
| `vector_size` | Length of each word vector |
| `window` | How many words on each side count as context |
| `min_count` | Ignore words that appear fewer times than this |
| `workers` | Number of threads for training |
| `sg` | `0` = CBOW (the default), `1` = Skip-gram |

> **Reality check:** in the lab, this three-sentence corpus gave almost no useful similarity ("pizza" and "pasta" scored about −0.006). **Embeddings need much more text than a few sentences.**

We can turn gensim vectors into a PyTorch layer too:

```python
vectors = torch.tensor(w2v.wv.vectors)                                      # one row per word
embedding = torch.nn.Embedding.from_pretrained(vectors, freeze=True)
word_to_index = {w: i for i, w in enumerate(w2v.wv.index_to_key)}
```

### 5.9 Using pretrained embeddings for classification

The lab (Part 2) uses frozen GloVe vectors for the news classifier from Part 3: the embedding layer is GloVe, so only the final linear layer learns. Settings in the lab: AG News data, 85% / 15% split, batch size 64, learning rate 0.1, 10 epochs, gradient clipping at 0.1.

```python
class GloveClassifier(nn.Module):
    def __init__(self, glove_vectors, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(glove_vectors, freeze=True, mode="mean")
        self.fc = nn.Linear(glove_vectors.shape[1], num_classes)
        self.fc.weight.data.uniform_(-0.5, 0.5)                             # set up ONLY the new layer
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        return self.fc(self.embedding(text, offsets))
```

Two things to get right:

1. **Do not overwrite the pretrained vectors.** In the lab's classifier, `init_weights` also sets `self.embedding.weight` to random numbers, **after** the GloVe vectors were loaded. That throws the pretrained knowledge away. We only initialize the new layer.
2. **The index of each word must match its row in the GloVe table.** One safe way is to use `glove.stoi` directly and add one extra all-zero row for unknown words:

```python
vectors = torch.cat([glove.vectors, torch.zeros(1, glove.vectors.shape[1])])     # last row = unknown word
unk_index = vectors.shape[0] - 1
text_pipeline = lambda text: [glove.stoi.get(w, unk_index) for w in tokenizer(text.lower())]
```

### 5.10 Part 5 summary

| Idea | One-sentence summary |
|---|---|
| **Word2Vec** | Learns word vectors so that words used in similar places are close |
| **Layers** | Input (vocabulary size) → embedding (our choice) → output (vocabulary size) |
| **CBOW** | Context words → target word |
| **Skip-gram** | Target word → one context word at a time |
| **Embeddings** | The trained weights of the embedding layer |
| **t-SNE and cosine similarity** | See and measure how close words are |
| **GloVe** | Pretrained vectors built from word co-occurrence counts |
| **Transfer learning** | Reuse (and often freeze) pretrained embeddings |

---

## Part 6: Sequence Models and RNNs

> **Lab for this part:** Developing a Sequence-to-Sequence Model (the RNN introduction).

### 6.1 Three kinds of sequence tasks

| Task type | Input → output | Example |
|---|---|---|
| **Sequence-to-sequence** | Many in → many out | **Machine translation**: an English sentence into a French sentence |
| **Sequence-to-label** | Many in → one label | **Document classification**: an article into a topic |
| **Label-to-sequence** | One in → many out | **Generative models**, such as creating an image from a single input |

Sequence-to-sequence models are also used for summarizing text and for chatbots.

### 6.2 Why do we need memory?

In a bag of words, "dog bites man" and "man bites dog" look the same, because the order is lost. For many tasks, the **order matters** and earlier words change the meaning of later words. A **recurrent neural network (RNN)** reads the sequence **one item at a time** and keeps a memory of what it has seen.

### 6.3 What is an RNN?

An **RNN** is a neural network for sequential or time-series data. It is designed to **remember past information and use it to influence what comes next**.

At each step $t$ it updates its memory, called the **hidden state** $h_t$:

$$h_t = \tanh(W_x \, x_t + W_h \, h_{t-1} + b)$$

| Symbol | Meaning |
|---|---|
| $x_t$ | The input at step $t$ (for example, one word's embedding) |
| $h_{t-1}$ | The memory from the previous step |
| $h_t$ | The new memory |
| $W_x$, $W_h$, $b$ | Learnable weights and bias |

If we need an output at each step, we add a layer on top: $y_t = \text{softmax}(W_o h_t + b_o)$.

**A tiny example from the lab.** Use $W_x = -10$, $W_h = 10$, $b = 0$ and the input sequence `1, 1, -1, -1, 1, 1`, starting with memory $h = 0$:

| Step | Input $x_t$ | Previous memory | $\tanh(-10 \cdot x_t + 10 \cdot h_{t-1})$ | New memory $h_t$ |
|---:|---:|---:|---|---:|
| 1 | 1 | 0 | tanh(−10) | **−1** |
| 2 | 1 | −1 | tanh(−20) | **−1** |
| 3 | −1 | −1 | tanh(0) | **0** |
| 4 | −1 | 0 | tanh(10) | **1** |
| 5 | 1 | 1 | tanh(0) | **0** |
| 6 | 1 | 0 | tanh(−10) | **−1** |

Look at the input `1`: at step 1 it gives −1, at step 5 it gives 0. **The same input gives different results because of the memory.** That is the whole point of an RNN. In real models, the weights are not chosen by hand: they are learned from data.

```python
W_xh, W_hh, b_h = torch.tensor(-10.0), torch.tensor(10.0), torch.tensor(0.0)
h = torch.tensor(0.0)
for x in [1, 1, -1, -1, 1, 1]:
    h = torch.tanh(x * W_xh + h * W_hh + b_h)
    print(round(h.item()))                       # -1, -1, 0, 1, 0, -1
```

### 6.4 Limits of RNNs, and LSTM and GRU

Plain RNNs have two problems:

- They remember only **short-term** information. Over many steps, the signal fades.
- They are **hard to train** (the gradients become tiny over long sequences, the vanishing gradient problem).

Two popular improvements fix this by adding **gates**, which are small switches that decide what to keep and what to forget.

| | **LSTM** (Long Short-Term Memory) | **GRU** (Gated Recurrent Unit) |
|---|---|---|
| **Gates** | Three: **input** (what new information to store), **forget** (what old information to throw away), **output** (what to pass on) | Two: **update** and **reset** |
| **Memory** | A hidden state plus a separate "memory belt" (the cell state) | Only a hidden state |
| **Size and speed** | More parameters, slower | Fewer parameters, faster |
| **PyTorch** | `nn.LSTM` | `nn.GRU` |

The translation lab uses an **LSTM**.

### 6.5 Part 6 summary

| Idea | One-sentence summary |
|---|---|
| **Sequence-to-sequence / sequence-to-label / label-to-sequence** | Many to many / many to one / one to many |
| **RNN** | Reads one item at a time and carries a hidden state as memory |
| **Limits** | Short memory and hard to train |
| **LSTM and GRU** | RNNs with gates that manage memory better |

---

## Part 7: Encoder-Decoder Models for Translation

> **Lab for this part:** Developing a Sequence-to-Sequence Model (German to English translation).

### 7.1 The idea

RNNs can build **sequence-to-sequence models**: one sequence goes in, another comes out. But the input and the output do not have the same length ("Das ist schön" has three words, "this is beautiful" also has three, but in general they differ). The **encoder-decoder** design solves this by splitting the job in two:

```text
Das  ist  schön   →  [ ENCODER ]  →  context  →  [ DECODER ]  →  this  is  beautiful
(input sequence)     reads the input    (summary)   writes the output   (output sequence)
```

| Part | Job |
|---|---|
| **Encoder** | A chain of RNNs that reads the input **one token at a time**, passing its hidden state from one RNN to the next. The **last hidden state** is called the **context**, and it goes to the decoder |
| **Decoder** | Another chain of RNNs that writes the translation **one token at a time**, starting from the context. It is **autoregressive**: each new token is made using the tokens it has already produced |

The lab says that RNN-based sequence-to-sequence models were introduced by Sutskever and colleagues in 2014. They are used for translation, question answering (chatbots), and summarization.

### 7.2 The encoder

The encoder turns each input token into an **embedded vector** with an embedding layer. That vector goes through the RNN cell (here an LSTM) to produce the hidden state. After the last token, we keep the final hidden state and cell state as the context.

```python
class Encoder(nn.Module):
    def __init__(self, vocab_len, emb_dim, hid_dim, n_layers, dropout_prob):
        super().__init__()
        self.hid_dim, self.n_layers = hid_dim, n_layers
        self.embedding = nn.Embedding(vocab_len, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_batch):                          # input_batch: [source length, batch size]
        embed = self.dropout(self.embedding(input_batch))    # [source length, batch size, embedding size]
        outputs, (hidden, cell) = self.lstm(embed)
        return hidden, cell                                  # the context for the decoder
```

### 7.3 The decoder

The decoder works step by step:

1. Its first input is the **`<bos>`** token (beginning of sentence). Its first memory is the encoder's context.
2. It embeds the input token, passes it through the LSTM, and a linear layer gives **one score for every word in the target vocabulary**.
3. The most likely word becomes the **next input** and part of the translation.
4. We stop when it produces **`<eos>`** (end of sentence).

```python
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim, self.hid_dim, self.n_layers = output_dim, hid_dim, n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):                  # input: [batch size], one token for each sentence
        embedded = self.dropout(self.embedding(input.unsqueeze(0)))      # [1, batch size, embedding size]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))          # [batch size, output vocabulary size]
        return prediction, hidden, cell
```

### 7.4 Connecting them: the Seq2Seq model and teacher forcing

The `Seq2Seq` class joins the encoder and the decoder. The encoder reads the source sentence. Then a loop runs the decoder once for every target position.

**Teacher forcing.** While training, we know the correct translation. At each step, we have a choice for the next decoder input:

| Choice | The next input is | Effect |
|---|---|---|
| **Teacher forcing** | The **correct** word from the target | Learns faster and more stably, because early mistakes do not pile up |
| **No teacher forcing** | The word the model **predicted** | Matches what happens when we really translate |

We pick at random with a probability called the **teacher forcing ratio** (0.5 in the lab, so half of the time). When we evaluate, the ratio is **0**.

```python
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device
        assert encoder.hid_dim == decoder.hid_dim, "Hidden sizes must match!"
        assert encoder.n_layers == decoder.n_layers, "Number of layers must match!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):          # src: [source length, batch], trg: [target length, batch]
        trg_len, batch_size = trg.shape
        outputs = torch.zeros(trg_len, batch_size, self.decoder.output_dim).to(self.device)
        hidden, cell = self.encoder(src)                             # the context
        input = trg[0, :]                                            # the first decoder input is <bos>
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            use_teacher = random.random() < teacher_forcing_ratio
            input = trg[t] if use_teacher else output.argmax(1)      # correct word, or the model's own guess
        return outputs                                               # [target length, batch size, target vocabulary size]
```

### 7.5 The data

The lab uses **Multi30k**: short image descriptions written in German and in English. A helper file prepares the data: it tokenizes the text, turns tokens into numbers, adds `<bos>` and `<eos>`, and pads the sentences in a batch to the same length.

- Tensors have the shape **[sequence length, batch size]**: the sequence comes first, because PyTorch's LSTM expects that by default. Each column is one sentence.
- A small example from the lab: the source tokens `0, 3, 4, 2, 1` mean `<bos>`, "Das", "ist", "schön", `<eos>`, and the target tokens `0, 2, 3, 5, 1` mean `<bos>`, "this", "is", "beautiful", `<eos>`.
- The helper can **reverse the source sentence** (`flip=True`). The lab says this can help the LSTM train.

### 7.6 Training

#### The settings used in the lab

| Setting | Value |
|---|---|
| Embedding size (encoder and decoder) | 128 |
| Hidden size | 256 |
| Number of LSTM layers | 1 |
| Dropout | 0.3 |
| Starting weights | Uniform random between −0.08 and 0.08 |
| Optimizer | Adam |
| Loss | Cross-entropy that **ignores the padding** (`ignore_index=<pad>`) |
| Gradient clipping | 1 |
| Teacher forcing | 0.5 while training, 0 while evaluating |
| Random seed | 1234 |

> **Note:** with only one LSTM layer, the LSTM's own dropout option has no effect (PyTorch warns about this). The dropout on the embeddings still works.

#### The training steps

The **objective** is to minimize the **cross-entropy loss** between the predicted words and the real words.

| Step | What we do |
|---:|---|
| 1 | Put the model in **training mode** (`model.train()`), which switches on layers such as dropout |
| 2 | Go through the data **batch by batch**; move the source (`src`) and target (`trg`) to the right device (CPU or GPU) |
| 3 | **Generate the predictions** by running the model |
| 4 | **Reshape** the output: drop the first position (`<bos>`) and flatten, so that every row is one predicted word and every column is a vocabulary score. The target is flattened the same way |
| 5 | **Calculate the loss**, run backpropagation, clip the gradients, and update the weights |
| 6 | After all batches, **average the loss** per batch |

```python
def train(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)                                    # [target length, batch, vocabulary]
        output = output[1:].view(-1, output.shape[-1])              # [(target length - 1) * batch, vocabulary]
        trg = trg[1:].contiguous().view(-1)                         # [(target length - 1) * batch]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)    # keep the gradients from exploding
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)
```

The evaluation function is the same, except for three things: `model.eval()`, `torch.no_grad()` (no learning), and **teacher forcing turned off** (`model(src, trg, 0)`). We save the model whenever the **validation loss** is the best so far.

```python
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters())
```

### 7.7 Translating a new sentence

Translating needs its own function, because we no longer have the correct answer. We encode the source sentence, start with `<bos>`, and let the decoder feed itself, always choosing the most likely word (this is called **greedy** decoding).

```python
def translate(model, src_tensor, trg_vocab, max_len=50):             # src_tensor: [source length, 1]
    model.eval()
    bos, eos = trg_vocab["<bos>"], trg_vocab["<eos>"]
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
        tokens = [bos]
        for _ in range(max_len):
            last = torch.tensor([tokens[-1]], device=src_tensor.device)
            prediction, hidden, cell = model.decoder(last, hidden, cell)
            next_token = prediction.argmax(1).item()
            if next_token == eos:                                    # the model says the sentence is finished
                break
            tokens.append(next_token)
    return " ".join(trg_vocab.get_itos()[t] for t in tokens[1:])
```

The lab's example translates the German sentence "Ein asiatischer Mann kehrt den Gehweg." (the reference English translation is "Asian man sweeping the walkway.").

### 7.8 Why are sequence-to-sequence models harder to train?

The course says they are harder to train than a simple RNN. Common reasons:

| Reason | In simple words |
|---|---|
| **Two networks** | The encoder and the decoder must learn together |
| **One token at a time** | A mistake early in the sentence changes everything after it |
| **A squeezed summary** | The whole input is squeezed into one context vector |
| **Different lengths** | We need padding, special tokens, and careful reshaping |
| **Training vs. translating** | Teacher forcing helps training, but at translation time the model must use its own guesses |

### 7.9 Part 7 summary

| Idea | One-sentence summary |
|---|---|
| **Encoder-decoder** | Read the input, summarize it as a context, then write the output |
| **Context** | The encoder's last hidden state, passed to the decoder |
| **Autoregressive** | The decoder builds the output one token at a time, using its previous tokens |
| **Teacher forcing** | Sometimes feed the correct word instead of the model's guess during training |
| **Training** | Cross-entropy, ignoring padding, gradient clipping, and reshaping the output |
| **Translating** | Encode, start at `<bos>`, repeat the decoder until `<eos>` |

---

## Part 8: Evaluating Language and Translation Models

> **Labs for this part:** Feed Forward Neural Networks (perplexity) and Developing a Sequence-to-Sequence Model (perplexity and BLEU).

### 8.1 Which score for which job?

| Score | What it tells us | Used for | Better when |
|---|---|---|---|
| **Loss** | How wrong the model is | Training any model | Lower |
| **Accuracy** | How often the top guess is exactly right | Classification | Higher |
| **Perplexity** | How uncertain the model is about the next word | Language models, translation models | Lower |
| **Precision, recall, F1** | How accurate and how complete the generated text is | Translation | Higher |
| **BLEU** | How close a translation is to reference translations | Translation | Higher |

### 8.2 Perplexity

**Perplexity** is a precious tool for judging the efficiency of language models and generative AI models. It is built from the **cross-entropy loss**, which measures how far the predicted word probabilities are from the real ones. We just take the **exponent of the loss**:

$$\text{Perplexity} = e^{\text{loss}}$$

**How to read it:** a model with perplexity 8 is about as unsure as if it had to choose among 8 equally likely words at each step. **Lower is better.**

| Average loss | Perplexity | Meaning |
|---:|---:|---|
| 0 | 1 | Always certain and right |
| 2.0 | 7.4 | Like choosing among about 7 words |
| 4.6 | 99.5 | Like choosing among about 100 words |

```python
import math
perplexity = math.exp(average_loss)            # use the average loss per token
```

In the labs, the perplexity curves go **down** while the loss goes down.

**Limits:** perplexity gives an overall measure of model performance, but it **does not capture the nuances of the quality of generated text** (is it fluent, is it correct, is it useful?). That is why we also use other scores.

### 8.3 Precision, recall, and F1 for translation

We can compare the words of the generated translation with the words of a correct (reference) translation.

| Score | Question it answers | Formula |
|---|---|---|
| **Precision** | Of the words we generated, how many are correct? (accuracy of the translation) | shared words ÷ words in the generated text |
| **Recall** | Of the words in the reference, how many did we produce? (completeness of the translation) | shared words ÷ words in the reference |
| **F1 score** | One number that balances the two (the harmonic mean) | 2 × precision × recall ÷ (precision + recall) |

**Example.**

- Reference: `the cat sat on the mat` (6 words)
- Generated: `the cat is on the mat` (6 words)
- Shared words: the, cat, on, the, mat = **5**

Precision = 5/6 = 0.83, recall = 5/6 = 0.83, F1 = 0.83.

```python
from collections import Counter

def overlap_scores(candidate, reference):
    cand, ref = candidate.split(), reference.split()
    common = sum((Counter(cand) & Counter(ref)).values())        # shared words
    precision, recall = common / len(cand), common / len(ref)
    f1 = 0 if common == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1

print(overlap_scores("the cat is on the mat", "the cat sat on the mat"))      # about (0.83, 0.83, 0.83)
```

Notice what happens if the output is **too short** (high precision, low recall) or **too long** (low precision, high recall). F1 punishes both. These scores only count words and ignore their order, which leads to BLEU.

### 8.4 BLEU score

**BLEU** (Bilingual Evaluation Understudy) compares a generated translation with **one or more reference translations**. It is useful because a sentence often has several good translations: we can put all of them in the reference list.

BLEU has three parts:

| Part | What it does |
|---|---|
| **N-gram precision** | Counts how many 1-word, 2-word, 3-word, and 4-word groups of the output also appear in a reference. Each group is counted **at most as many times as it appears in the reference** ("clipping"), so repeating a word does not help |
| **Brevity penalty** | Punishes outputs that are too short. If the output length $c$ is longer than the reference length $r$, the penalty is 1. Otherwise it is $e^{1 - r/c}$ |
| **Geometric mean** | Combines the four precisions (usually with equal weights) and multiplies by the brevity penalty |

$$\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right), \qquad w_n = \tfrac{1}{N}$$

BLEU is between 0 and 1. **Higher is better.**

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

references = ["asian man sweeping the walkway .".split(),
              "an asian man is sweeping the sidewalk .".split()]
hypothesis = "an asian man is sweeping the walkway .".split()

smooth = SmoothingFunction().method1
print(sentence_bleu(references, hypothesis, smoothing_function=smooth))      # about 0.95
```

**Tips:**

- **Make the text consistent.** BLEU compares exact tokens, so use the same lowercase and tokenization for the output and the references.
- **Use smoothing for short sentences.** If a sentence has no matching 4-word group, the score collapses to almost 0 (for the two-word output "man walking" the unsmoothed score is about 0, and the smoothed score is 0.02). The smoothing function avoids this.
- A single sentence score is noisy. For a real evaluation, average over many sentences.

### 8.5 Libraries

In NLP, popular libraries already implement the evaluation metrics, so we do not need to write them ourselves:

| Library | What we use it for |
|---|---|
| **NLTK** | `sentence_bleu` for BLEU; `FreqDist`, `bigrams`, and tokenizers for the counting models |
| **PyTorch** | Loss functions (`CrossEntropyLoss`), from which we compute perplexity |

### 8.6 Part 8 summary

| Idea | One-sentence summary |
|---|---|
| **Loss** | The main number to watch while training |
| **Perplexity** | The exponent of the loss; lower means a more confident, better model |
| **Precision and recall** | Accuracy and completeness of the generated words |
| **F1** | The harmonic mean of precision and recall |
| **BLEU** | N-gram overlap with references, with a penalty for short outputs |

---

## Part 9: Putting It All Together

### 9.1 The big map

| Problem | Input → output | Model | Loss | How we score it |
|---|---|---|---|---|
| **Document classification** | Text → one category | EmbeddingBag + Linear layer | Cross-entropy | Accuracy |
| **Next-word prediction** | Previous words → the next word | Embeddings + hidden layer + output layer (neural n-gram) | Cross-entropy | Loss and perplexity |
| **Word embeddings** | A word ↔ its neighbors | CBOW or Skip-gram | Cross-entropy | Plots and cosine similarity |
| **Translation** | Sentence → sentence | Encoder-decoder LSTM | Cross-entropy (padding ignored) | Perplexity and BLEU |

And the quick "what do we use when" list:

| We need to... | We use... |
|---|---|
| Give every word a vector | `nn.Embedding` |
| Give every document a vector | `nn.EmbeddingBag` (with offsets) |
| Choose a class or a word | The logits, then `argmax`, and `nn.CrossEntropyLoss` (raw logits in) |
| Predict the next word | A neural n-gram model |
| Learn word vectors from our own text | CBOW or Skip-gram (or `gensim`) |
| Reuse knowledge from huge text | Pretrained GloVe, frozen (`freeze=True`) |
| Remember the order in a sequence | RNN, and LSTM or GRU for longer memory |
| Turn one sequence into another | Encoder-decoder with teacher forcing |
| Judge a language model | Loss and perplexity |
| Judge a translation | BLEU (and precision, recall, F1) |

### 9.2 The five labs at a glance

| Lab | What we do | What to remember |
|---|---|---|
| **Feed Forward Neural Networks** | Tokenize and index a small text, build embeddings, make (context, target) pairs, build a batch function, train a neural n-gram model with context 2, 4, and 8, plot loss and perplexity; exercise with nursery rhymes | A bigger context lowers the training loss, but on a tiny text it mostly means memorizing |
| **Language Modelling** | Count words with NLTK; build unigram, bigram, and trigram models; generate text | Counting models are simple, but they have no meaning and they repeat themselves |
| **Integrating Word2Vec, Part 1** | Build CBOW and Skip-gram models on a toy text; train; plot embeddings with t-SNE | Keep the embedding weights; tiny data gives rough results |
| **Integrating Word2Vec, Part 2** | Load GloVe, train a gensim Word2Vec model, classify AG News with pretrained embeddings | Do not overwrite pretrained vectors; this lab needs an internet connection to download GloVe |
| **Developing a Sequence-to-Sequence Model** | An RNN example, then the encoder, the decoder, the Seq2Seq model, training, translating German to English, perplexity, and BLEU | Teacher forcing in training, off in evaluation; greedy decoding to translate |

### 9.3 Watch-outs found in the labs

We noticed these while reading the lab notebooks. None of them stops us from learning the ideas, but it is good to know them.

| Where | Watch-out | What to do |
|---|---|---|
| **Feed Forward lab** | The loss gets lower with a bigger context, but the text is tiny (79 different words) and there is no validation set, so the model mostly **memorizes** | Check the loss and perplexity on **held-out text** before trusting a bigger context |
| **Word2Vec, Part 1** | The settings (learning rate 5, 400 epochs) are for a toy demo | For real data use more text and a smaller learning rate |
| **Word2Vec, Part 2** | The classifier's `init_weights` **replaces the pretrained GloVe vectors with random numbers** (after loading them) | Initialize only the new layer (Section 5.9) |
| **Word2Vec, Part 2** | The saved notebook shows **download errors** for GloVe, so it has no results; and a comment says 5% validation while the code uses 15% | Run it with internet access, and trust the code over the comment |
| **Word2Vec, Part 2** | The gensim example has only three sentences, so the similarities are meaningless ("pizza" vs. "pasta" about −0.006) | Use a much larger text |
| **Sequence-to-sequence lab** | With one LSTM layer, the LSTM's `dropout` option does nothing and PyTorch warns | Use two or more layers, or rely on the embedding dropout |
| **Sequence-to-sequence lab** | BLEU compares exact tokens and gives almost 0 for short outputs | Use the same case in both texts, and add smoothing |

### 9.4 Glossary

| Word | Simple meaning |
|---|---|
| **Token** | A small piece of text, usually a word |
| **Vocabulary** | The list of all tokens, each with its own number |
| **Embedding** | A short list of learned numbers that stands for a word |
| **Logits** | The raw scores a network gives before turning them into probabilities |
| **Argmax** | The position of the biggest number |
| **Loss** | One number for how wrong the model is |
| **Cross-entropy** | A loss that punishes a low probability on the right answer |
| **Epoch** | One full pass through the training data |
| **Batch** | A small group of samples used for one update |
| **Hyperparameter** | A setting we choose before training |
| **Language model** | A model that predicts the next word |
| **Context** | The words we use as clues |
| **N-gram** | A group of N words in a row |
| **CBOW** | Context words predict the target word |
| **Skip-gram** | The target word predicts a context word |
| **RNN** | A network that reads items one by one and keeps a hidden state as memory |
| **LSTM / GRU** | RNNs with gates that manage memory |
| **Encoder / decoder** | The part that reads the input / the part that writes the output |
| **Context vector** | The encoder's last hidden state, passed to the decoder |
| **Autoregressive** | Producing one token at a time, using the earlier tokens |
| **Teacher forcing** | Feeding the correct word instead of the model's guess during training |
| **Perplexity** | The exponent of the loss; lower is better |
| **BLEU** | A score for how close a translation is to the reference translations |

### 9.5 Self-check

Try each question first, then open the answer.

**1. What is the difference between a one-hot vector and an embedding?**

<details>
<summary>Show answer</summary>

A one-hot vector is long, has a single 1, and does not change. An embedding is a short list of learned numbers, and similar words can end up close together.

</details>

**2. What does `nn.EmbeddingBag` return, and what are offsets for?**

<details>
<summary>Show answer</summary>

It returns one vector per document (the average of its word vectors by default). The offsets tell it where each document starts inside the long joined list of word indexes.

</details>

**3. What is the difference between learnable parameters and hyperparameters? Give one example of each.**

<details>
<summary>Show answer</summary>

Learnable parameters are changed by training (for example, the weights or the embeddings). Hyperparameters are chosen by us before training (for example, the learning rate or the batch size).

</details>

**4. Why do we give raw logits to `CrossEntropyLoss`?**

<details>
<summary>Show answer</summary>

`CrossEntropyLoss` already includes the softmax step. Giving it probabilities would apply softmax twice.

</details>

**5. Where does Monte Carlo sampling appear in training?**

<details>
<summary>Show answer</summary>

The true loss is an average over all possible text, which we cannot compute. We estimate it by averaging the loss over the samples in each batch.

</details>

**6. What are the training, validation, and test sets for?**

<details>
<summary>Show answer</summary>

Training: learning the weights. Validation: tuning hyperparameters and watching progress. Test: a final honest score of real-world performance.

</details>

**7. Why do we prefer the loss to accuracy as the KPI for language models?**

<details>
<summary>Show answer</summary>

Accuracy only counts exact top guesses, but many next words can be reasonable. The loss looks at how much probability the right word received, so it improves smoothly, even when the top guess is still wrong.

</details>

**8. In "I like dogs and I kinda like cats", what does a bigram model predict after "like"?**

<details>
<summary>Show answer</summary>

"dogs" and "cats" are tied, each with probability 1/2.

</details>

**9. A neural n-gram model has embedding size 10 and context size 3. How many numbers go into its hidden layer?**

<details>
<summary>Show answer</summary>

30 (3 words × 10 numbers each, joined side by side).

</details>

**10. Which model predicts the target word from its context words: CBOW or Skip-gram? Which one is usually better for rare words?**

<details>
<summary>Show answer</summary>

CBOW predicts the target from the context. Skip-gram is usually better for rare words and small datasets.

</details>

**11. After training a Word2Vec model, what do we keep? What does `freeze=True` do with pretrained embeddings?**

<details>
<summary>Show answer</summary>

We keep the weights of the embedding layer: they are the word vectors. `freeze=True` stops those weights from being updated during training.

</details>

**12. In the tiny RNN example, why does the input 1 sometimes give −1 and sometimes 0?**

<details>
<summary>Show answer</summary>

Because of the memory. The new hidden state depends on the previous hidden state as well as on the input.

</details>

**13. What is the context in an encoder-decoder model, and what is teacher forcing?**

<details>
<summary>Show answer</summary>

The context is the encoder's last hidden state (and cell state), which is passed to the decoder. Teacher forcing means feeding the decoder the correct word instead of its own guess during training. It is turned off (ratio 0) when we evaluate.

</details>

**14. A model has an average loss of 2.0. What is its perplexity?**

<details>
<summary>Show answer</summary>

e² ≈ 7.4. It is about as unsure as choosing among 7 equally likely words.

</details>

**15. Why does BLEU have a brevity penalty?**

<details>
<summary>Show answer</summary>

Short outputs can have a very high n-gram precision (for example, a single correct word). The brevity penalty lowers the score of outputs that are shorter than the reference.

</details>
