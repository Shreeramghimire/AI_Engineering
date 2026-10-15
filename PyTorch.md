# PyTorch (From Linear Regression to Deep Neural Networks and CNNs)

> **How to use this guide:** the guide has five stages that build on each other, so read them in order the first time. Each stage opens with its goals, each part ends with a "How the pieces connect" note that links it to what comes next, and each stage closes with a short self-check (the answers are folded away, so try the question first). Use the **Quick Reference** at the end for revision.

## Roadmap

**Stage 1: Foundations**

| Part | Topic | Key question it answers |
|---|---|---|
| [1](#part-1--pytorch-foundations) | PyTorch foundations | What is PyTorch, and what are tensors and autograd? |

**Stage 2: Predicting Numbers and Categories**

| Part | Topic | Key question it answers |
|---|---|---|
| [2](#part-2--regression-predicting-a-number) | Regression | How do we train a model to predict a number? |
| [3](#part-3--binary-classification-predicting-a-category) | Binary classification | How do we predict a category (yes/no)? |
| [4](#part-4--probability-foundations-where-classification-losses-come-from) | Probability foundations | Where do classification losses come from? (Bernoulli, likelihood, MLE) |
| [5](#part-5--loss-functions-for-classification) | Loss functions for classification | Why cross-entropy and not MSE? |
| [6](#part-6--logistic-regression-in-pytorch-spam-detection) | Logistic regression in PyTorch | How do Parts 3 to 5 look as working code? |
| [7](#part-7--multi-class-classification-with-softmax) | Multi-class classification | How do we handle more than two classes? (Softmax) |

**Stage 3: Neural Networks**

| Part | Topic | Key question it answers |
|---|---|---|
| [8](#part-8--neural-networks-with-hidden-layers-mnist) | Neural networks with hidden layers | How do dense layers and forward propagation work on a real example (MNIST)? |
| [9](#part-9--deep-neural-networks) | Deep neural networks | What changes when we add more hidden layers, and how do we build them? |
| [10](#part-10--backpropagation-and-the-vanishing-gradient-problem) | Backpropagation and vanishing gradients | How are gradients computed, and why can they fade in deep networks? |

**Stage 4: Training Deep Networks Well**

| Part | Topic | Key question it answers |
|---|---|---|
| [11](#part-11--weight-initialization-and-batch-normalization) | Weight initialization and batch normalization | How do we start, and keep, activations in a healthy range? |
| [12](#part-12--better-training-momentum-optimizers-and-learning-rate-strategies) | Better training: momentum, optimizers, learning rates | Which momentum, optimizer, and learning-rate strategies make training faster and more stable? |
| [13](#part-13--overfitting-and-dropout) | Overfitting and dropout | How do we stop a deep network from memorizing the training data? |

**Stage 5: Images and Modern Practice**

| Part | Topic | Key question it answers |
|---|---|---|
| [14](#part-14--convolutional-neural-networks-cnns) | Convolutional neural networks | How do we process images while respecting their spatial structure? |
| [15](#part-15--resnets-and-transfer-learning) | ResNets and transfer learning | How do we train very deep networks and reuse pretrained models? |
| [16](#part-16--training-at-scale-and-deployment) | Training at scale and deployment | How do we use GPUs, reduce boilerplate, and ship a model? |

**Finish with:** [Quick Reference](#quick-reference), a one-page summary of what to use when.

---

# Stage 1: Foundations

> **Part 1.** By the end of this stage we can:
> - explain what PyTorch is and why we use it even for simple models
> - create and use tensors
> - compute gradients automatically with autograd

## Part 1 PyTorch Foundations

### 1.1 What is PyTorch?

**PyTorch** is a Python library for building and training machine learning models.

### 1.2 Why use PyTorch for something as simple as linear regression?

For simple linear regression, we could use Excel or plain math. But PyTorch gives:

- Fast math on lots of numbers, especially using GPUs
- Automatic gradients: it figures out how to adjust weights for you
- Efficient handling of big datasets

### 1.3 Two big ideas

#### Idea 1: Tensors

Tensors are like NumPy arrays, but they can run on GPUs.

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
```

#### Idea 2: Autograd

PyTorch automatically computes gradients.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # tensor(4.)  ← dy/dx = 2x = 4
```

---

## Stage 1 self-check

Try each question first, then open the answer.

**1. What do tensors and autograd each give us?**

<details>
<summary>Show answer</summary>

Tensors are arrays (like NumPy arrays) that can run on GPUs. Autograd automatically computes gradients, so we never derive them by hand.

</details>

**2. For `y = x**2` with `x = 2.0` (`requires_grad=True`), what does `x.grad` hold after `y.backward()`?**

<details>
<summary>Show answer</summary>

`tensor(4.)`, because dy/dx = 2x = 4.

</details>

---

# Stage 2: Predicting Numbers and Categories

> **Parts 2 to 7.** By the end of this stage we can:
> - train a model with the 5-step training loop
> - choose the output layer and loss for regression, binary classification, and multi-class classification
> - explain why cross-entropy (maximum likelihood), not MSE, is the loss for classification
> - turn scores into probabilities with sigmoid and softmax

## Part 2 Regression: Predicting a Number

### 2.1 Linear regression: the core idea

Linear regression finds the best line:

$$y = wx + b$$

| Symbol | Meaning | Example |
|---|---|---|
| $y$ | What we predict | Ice cream sales |
| $x$ | Input | Temperature |
| $w$ | Weight / slope | |
| $b$ | Bias / intercept | |

**Training** = finding the best $w$ and $b$ so the line fits the data.

### 2.2 Why learn it in PyTorch?

- It's the "Hello World" of ML.
- The same workflow applies to deep neural networks — just swap in a bigger model.
- You learn tensors, `nn.Module`, loss functions, optimizers, and the training loop, all of which transfer to CNNs, RNNs, and Transformers.

### 2.3 The training loop (5 steps)

Every model in this guide is trained with the same loop:

```text
Forward → Loss → Reset → Backward → Step → Repeat
```

| # | Step | What it does | Code |
|---|---|---|---|
| 1 | **Forward** | Make a prediction | `y_pred = model(X)` |
| 2 | **Loss** | Measure how wrong the prediction is | `loss = criterion(y_pred, y)` |
| 3 | **Reset** | Clear old gradients | `optimizer.zero_grad()` |
| 4 | **Backward** | Compute new gradients | `loss.backward()` |
| 5 | **Step** | Update the weights | `optimizer.step()` |

> **Note:** PyTorch *accumulates* gradients by default, so they must be cleared before each `backward()` call. Whether `zero_grad()` is placed at the start of the loop or just before `backward()`, the result is the same. All code in this guide uses the order shown above.

### 2.4 Multi-output linear regression

#### What is it?

**Single-output** linear regression predicts **one number**:

$$y = wx + b$$

**Multi-output** linear regression predicts **several numbers at once**:

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

**Example:** predict both the **weight** and the **sweetness** of a cake from its ingredients → two outputs.

The model doesn't give just one answer; it gives a **vector of answers**, all at once.

#### Analogy: baking a cake

| Concept | Analogy |
|---|---|
| Input features | Ingredients you put in |
| Output 1 (weight) | How much the cake weighs |
| Output 2 (sweetness) | How sweet the cake tastes |
| Weight matrix $W$ | Table showing how each ingredient affects each outcome |
| Bias vector $\mathbf{b}$ | Little extra nudge to make each prediction more accurate |
| Training | Tasting the cake, comparing to the ideal, adjusting the recipe |

The model learns to perfect **all predictions together**, just like perfecting a recipe to get the best cake every time.

#### The math: vectors and shapes

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

Where:

- $\mathbf{x}$ = input vector (e.g., $[flour, sugar, eggs]$)
- $W$ = weight **matrix** (shape: `num_outputs × num_features`)
- $\mathbf{b}$ = bias **vector** (shape: `num_outputs`)
- $\mathbf{y}$ = output vector (e.g., $[weight, sweetness]$)

**Concrete example:**

- Inputs: 3 features (flour, sugar, eggs)
- Outputs: 2 targets (weight, sweetness)

Then:

- $W$ is a **2 × 3 matrix**
- $\mathbf{b}$ is a **2-element vector**
- $\mathbf{y}$ is a **2-element vector**

$$
W =
\begin{bmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23}
\end{bmatrix},
\qquad
\mathbf{b} =
\begin{bmatrix}
b_1 \\
b_2
\end{bmatrix}
$$

Each **row** of $W$ corresponds to one **output**. Each **column** corresponds to one **input feature**.

> **Note on shapes:** this `(outputs × features)` layout is exactly how PyTorch stores it: `nn.Linear(in_features=3, out_features=2).weight` has shape `(2, 3)`. The softmax section (Part 7) uses the same convention.

### 2.5 The cost function

For single-output MSE (mean squared error):

$$\text{Loss} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

For multi-output, we **sum the squared errors across all outputs**:

$$\text{Loss} = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M}(y_{ij} - \hat{y}_{ij})^2$$

Where:

- $N$ = number of samples
- $M$ = number of outputs

**Key idea:** The model improves **all predictions together** — one aggregated loss drives learning for every output.

> **Note:** by default, `nn.MSELoss()` averages over *all* $N \times M$ values, so it equals the formula above divided by $M$. The learning direction is the same; only the scale differs.

### 2.6 Multi-output regression in PyTorch

**The model** (3 input features → 2 outputs):

```python
import torch
import torch.nn as nn

model = nn.Linear(in_features=3, out_features=2)
```

**The loss:**

```python
criterion = nn.MSELoss()
```

**Complete training example:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

#  1. Data 
# Inputs: [flour, sugar, eggs]
X = torch.tensor([
    [2.0, 1.0, 3.0],
    [3.0, 2.0, 2.0],
    [1.0, 3.0, 1.0],
    [4.0, 1.0, 2.0],
    [2.0, 2.0, 3.0],
])

# Targets: [weight, sweetness]
y = torch.tensor([
    [250.0, 7.0],
    [300.0, 6.0],
    [200.0, 8.0],
    [350.0, 5.0],
    [280.0, 7.5],
])

# 2. Model 
model = nn.Linear(in_features=3, out_features=2)

# 3. Loss 
criterion = nn.MSELoss()

# 4. Optimizer 
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# 5. Training Loop 
for epoch in range(5000):
    y_pred = model(X)                 # Forward → shape (5, 2)
    loss = criterion(y_pred, y)       # Aggregated loss over both outputs
    optimizer.zero_grad()             # Reset gradients
    loss.backward()                   # Backward
    optimizer.step()                  # Update W and b

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 6. Predict 
model.eval()
with torch.no_grad():
    new_recipe = torch.tensor([[2.5, 1.5, 2.5]])
    prediction = model(new_recipe)
    print(f'Predicted [weight, sweetness]: {prediction.numpy()}')
```

> **What to expect:** the loss falls quickly at first and then keeps shrinking slowly. The targets are large (hundreds) and the learning rate is small, so the loss will still be well above zero after 5000 epochs. This is fine for a demo; scaling the data or raising the learning rate would speed things up.

---

## Part 3 Binary Classification: Predicting a Category

### 3.1 Linear classifiers and decision boundaries

Imagine we have fruits on a table: apples and oranges. We want to separate them automatically.

A **linear classifier** does exactly that: it draws a straight line that splits the fruits into two groups based on their features (like color or size).

- Fruit on one side → Apple
- Fruit on the other side → Orange

That line is called the **decision boundary**. A decision boundary is the line that separates classes:

| Dimensions | Boundary |
|---|---|
| 2D | A line |
| 3D | A plane |
| Higher dimensions | A hyperplane |

The classifier uses a simple formula to decide which side a point belongs to.

### 3.2 The linear function (the score)

Every linear classifier computes a **score**:

$$z = \mathbf{w} \cdot \mathbf{x} + b$$

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | Feature vector (e.g., color, size) |
| $\mathbf{w}$ | Weight vector (importance of each feature) |
| $b$ | Bias (shifts the boundary) |
| $z$ | Score used to decide the class |

**Example:** If $z > 0$ → Class 1 (apple). If $z \leq 0$ → Class 0 (orange).

### 3.3 From scores to class labels

The linear function gives a **continuous score** (any number). To get a class label, we apply a **threshold**:

```text
If z > 0 → Class 1
Else     → Class 0
```

**Problem:** this gives a hard yes/no answer, with no confidence.

- "This is an apple."
- But how sure are we?

### 3.4 Logistic regression and the sigmoid function

#### In plain words

Logistic regression is a way for a computer to make decisions about things that have two possible outcomes, like yes/no, true/false, or cat/dog. Imagine you want to teach a computer to decide if an email is spam. Logistic regression looks at the email and gives a probability, a number between 0 and 1, for how likely it is to be spam. Close to 1 means probably spam; close to 0 means probably not.

Think of it as a smart gatekeeper that uses a smooth curve (the **sigmoid function**) to decide how confident it is about each choice. Instead of making a hard yes/no decision right away, it gives a gentle "maybe" score. This smooth scoring matters because it lets the computer improve its guesses step by step, using feedback from its mistakes.

#### The math

Logistic regression passes the score $z$ through the **sigmoid function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \mathbf{w}^T\mathbf{x} + b$$

This converts any score into a **probability between 0 and 1**, the probability of belonging to class 1.

| $z$ | $\sigma(z)$ | Meaning |
|---|---|---|
| Large negative | Close to 0 | Very likely Class 0 |
| Zero | 0.5 | Uncertain |
| Large positive | Close to 1 | Very likely Class 1 |

**Example outputs:**

- 0.95 → "95% sure it's an apple"
- 0.60 → "60% sure it's an apple"
- 0.50 → "I have no idea"
- 0.10 → "90% sure it's an orange"

Much more informative than a simple yes/no.

### 3.5 Hard decision vs. soft probability

| | Linear classifier | Logistic regression |
|---|---|---|
| Output | Class label (0 or 1) | Probability (0 to 1) |
| Confidence | No | Yes |
| Boundary | The line $z = 0$ | The same line $z = 0$, but the probability fades smoothly across it |
| Example | "Apple" | "85% apple" |

### 3.6 Applying a threshold

To convert a probability back into a class label, we pick a threshold (usually **0.5**):

```text
If probability ≥ 0.5 → Class 1
Else                 → Class 0
```

Because $\sigma(0) = 0.5$, thresholding the probability at 0.5 is the same as checking whether $z > 0$.

The threshold can be adjusted:

- Lower threshold → more Class 1 predictions (higher recall)
- Higher threshold → fewer Class 1 predictions (higher precision)

### 3.7 Worked example: email spam detection

We want to classify emails as spam (1) or not spam (0). Features might be:

- Number of links in the email
- Frequency of words like "free," "win," "urgent"
- Whether the sender is in your contacts

The model computes $z = 2.5$ for a particular email. Applying the sigmoid:

$$\sigma(2.5) = \frac{1}{1 + e^{-2.5}} \approx 0.924$$

**Interpretation:** there's a 92.4% probability this email is spam.

| Email | $z$ (raw score) | $\sigma(z)$ | Interpretation |
|---|---:|---:|---|
| A | 3.0 | 0.953 | 95.3% spam |
| B | 0.0 | 0.500 | 50% spam (uncertain) |
| C | −2.0 | 0.119 | 11.9% spam (likely not spam) |

**More real-world uses:**

- Medical diagnosis: probability a patient has a disease given symptoms
- Credit card fraud: probability a transaction is fraudulent
- Customer churn: probability a customer will cancel their subscription
- Ad click prediction: probability a user will click on an ad

---

## Part 4 Probability Foundations: Where Classification Losses Come From

Before we can pick a loss function for classification, we need a way to measure how well predicted probabilities explain the labels we actually observed. Bernoulli distributions, likelihood, and maximum likelihood estimation (MLE) give us that. They are the foundation of cross-entropy in Part 5.

### 4.1 Bernoulli distribution

A **Bernoulli distribution** models a single experiment with two outcomes:

- **Heads (success):** probability = $\theta$
- **Tails (failure):** probability = $1 - \theta$

Mathematically:

$$P(x) = \theta^x (1-\theta)^{1-x}$$

where $x \in \{0, 1\}$ (1 = heads, 0 = tails).

**Example:** if $\theta = 0.2$, then:

- $P(\text{heads}) = 0.2$
- $P(\text{tails}) = 0.8$

### 4.2 Likelihood of a sequence

Suppose we flip the coin 3 times and get **Heads, Heads, Tails** (H, H, T). Since flips are independent, we multiply the probabilities:

$$P(H,H,T) = \theta \cdot \theta \cdot (1-\theta) = \theta^2 (1-\theta)$$

General formula for $n$ flips with $k$ heads:

$$L(\theta) = \theta^{k} (1-\theta)^{n-k}$$

This $L(\theta)$ is called the **likelihood**; it tells us how probable the observed data is for a given $\theta$.

### 4.3 Why use log-likelihood?

Multiplying many small probabilities can underflow (become ~0), and the product is harder to differentiate. Take the **log**:

$$\log L(\theta) = k \log(\theta) + (n-k) \log(1-\theta)$$

This turns **multiplication into addition**, which is much easier to work with. Crucially, the $\theta$ that maximizes $\log L$ also maximizes $L$ (log is monotonic).

### 4.4 Maximum likelihood estimation (MLE)

We want the $\theta$ that makes the observed data most likely. Take the derivative and set it to 0:

$$\frac{d}{d\theta} \log L = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$$

Solving gives:

$$\boxed{\hat{\theta}_{MLE} = \frac{k}{n}}$$

**In simple words:** the MLE is just the fraction of heads we observed. If we flipped 10 times and got 7 heads, $\hat{\theta} = 0.7$.

### 4.5 PyTorch example: MLE two ways

```python
import torch

# Observed flips: 1 = heads, 0 = tails
flips = torch.tensor([1., 1., 0., 1., 0., 1., 1., 0., 1., 1.])
n = flips.numel()
k = flips.sum()

# Closed-form MLE 
theta_mle = k / n
print(f"Closed-form MLE: {theta_mle.item():.4f}")  # 0.7

# Via gradient descent (how PyTorch really works) 
# Use logit (unconstrained) so theta stays in (0,1)
logit = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([logit], lr=0.5)

for step in range(200):
    optimizer.zero_grad()
    theta = torch.sigmoid(logit)              # maps R -> (0,1)
    # Negative log-likelihood
    nll = -(k * torch.log(theta) + (n - k) * torch.log(1 - theta))
    nll.backward()
    optimizer.step()

print(f"Gradient-descent MLE: {torch.sigmoid(logit).item():.4f}")
```

---

## Part 5 Loss Functions for Classification

Imagine we're building a spam filter. For each email, our model outputs a **probability**:

- "90% sure this is spam" → prediction = 0.9
- "60% sure this is not spam" → prediction = 0.4 (since not-spam = 1 − 0.6)

But the true label is binary: **spam (1)** or **not spam (0)**.

So the question becomes: **how do we score the model's probability predictions against a 0/1 label?** That is exactly what a **loss function** does. We'll look at two poor choices first, then the right one.

### 5.1 First attempt: count the mistakes (0–1 loss)

Suppose we define the loss as:

$$
\text{Loss} =
\begin{cases}
0, & \text{if prediction is correct} \\
1, & \text{if prediction is wrong}
\end{cases}
$$

**Example: loan approval**

| Applicant | True label | Predicted probability | Error-count loss |
|---|---|---:|---|
| A | Approved (1) | 0.51 | 0 (rounded to 1, correct) |
| B | Approved (1) | 0.99 | 0 (correct) |
| C | Rejected (0) | 0.49 | 0 (rounded to 0, correct) |
| D | Rejected (0) | 0.01 | 0 (correct) |

Whether the model predicts 0.51 or 0.99 for applicant A, the error count is the same (0). There's no gradient telling the model "0.99 is better than 0.51," so the model can't improve.

Worse, if applicant A were predicted at 0.49 (wrong), the loss jumps to 1 — but the gradient is zero everywhere except at the decision boundary. Gradient descent has nothing to follow.

### 5.2 Second attempt: MSE — why it's a poor fit for classification

**Mean squared error (MSE)** measures how accurate a model is by calculating the average squared distance between its predictions and the actual results. Because it measures physical distance, it is the gold standard for **regression tasks** (predicting continuous numbers like house prices, temperature, or stock values) rather than **classification tasks**.

For one sample:

$$\text{MSE} = (y - \hat{y})^2$$

It looks reasonable, but there's a hidden problem: when you combine MSE with a **sigmoid** output (which is what binary classifiers use), the gradients become tiny.

- With **sigmoid + MSE**, the gradient of the loss with respect to the weights includes a factor $\sigma'(z) = \hat{y}(1-\hat{y})$.
- When the model is very wrong (e.g., $\hat{y} \approx 0$ but $y = 1$), $\sigma'(z) \approx 0$ → the **gradient vanishes** → learning stalls.
- This creates **flat regions** in the loss surface.

### 5.3 The right tool: cross-entropy loss

#### Intuition

Cross-entropy loss looks at the predicted probability of the right answer and uses logarithms to measure how close the prediction is to the truth. It measures how "surprised" the model is by the true label.

Imagine a guessing game where you have to predict whether a picture shows a cat. Cross-entropy loss is the scorekeeper:

- Confidently say "cat" when it really is a cat → good score (**low loss**).
- Confidently say "cat" when it's not → bad score (**high loss**).

It punishes wrong *and* overconfident guesses more harshly, which helps the model learn.

It is also **smooth**: like a gentle hill, it lets the model improve its guesses step by step by following the slope (gradients) downhill toward the best prediction.

- Cross-entropy measures how well predicted probabilities match true class labels, giving a **smooth and continuous loss surface**.
- This smoothness allows **gradient-based optimization** to update parameters efficiently **without stalling**.
- MSE, by contrast, can create flat regions when predictions are very wrong, so the model stops learning.

#### Formula (binary cross-entropy, BCE)

$$\text{BCE}(y, \hat{y}) = -\big[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \big]$$

where:

- $y$ = true label (0 or 1)
- $\hat{y}$ = predicted probability (between 0 and 1)

#### Seeing it in action

| True label $y$ | Predicted $\hat{y}$ | Loss $-\log(\hat{y})$ | Meaning |
|---|---|---|---|
| 1 (spam) | 0.9 | 0.105 | Small loss — good guess |
| 1 (spam) | 0.5 | 0.693 | Medium loss — uncertain |
| 1 (spam) | 0.1 | 2.303 | **Big loss** — confident but wrong! |

Notice the **confident-but-wrong penalty** — that's the key feature.

#### Why it fixes the MSE problem

With **sigmoid + cross-entropy**, the $\sigma'(z)$ term cancels out mathematically, giving clean gradients that don't vanish.

### 5.4 Cross-entropy is MLE in disguise

Maximum likelihood estimation (Part 4) says: find the parameters $\theta$ that make the observed data most probable.

For binary labels modeled as Bernoulli:

$$P(y \mid \hat{y}) = \hat{y}^y (1-\hat{y})^{1-y}$$

Take the **log** (easier to work with):

$$\log P(y \mid \hat{y}) = y \log(\hat{y}) + (1-y)\log(1-\hat{y})$$

Now **negate** it (because we *minimize* loss, but MLE *maximizes* likelihood):

$$\text{Loss} = -\big[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \big]$$

**That's exactly cross-entropy loss.** Minimizing BCE = maximizing likelihood. They are the same thing.

### 5.5 MSE vs. BCE in PyTorch

Let's compare both losses on a simple binary classification problem:

```python
import torch
import torch.nn as nn

# Data: 4 samples, 2 features 
X = torch.tensor([[1.0, 2.0],
                  [2.0, 1.0],
                  [-1.0, -2.0],
                  [-2.0, -1.0]])
y = torch.tensor([[1.0], [1.0], [0.0], [0.0]])  # binary labels

# A tiny model: linear layer + sigmoid 
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# Try both losses 
def train(loss_fn, name, lr=0.1, steps=200):
    # Reset the model
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    
    print(f"{name}: final loss = {loss.item():.4f}")
    return model

# Mean Squared Error
mse_model = train(nn.MSELoss(), "MSE")

# Binary Cross-Entropy
bce_model = train(nn.BCELoss(), "BCE")
```

**What we'll observe:**

- BCE converges faster and to a lower loss.
- MSE learns more slowly, especially when predictions start off very wrong.

### 5.6 Best practice: use `BCEWithLogitsLoss`

In practice, PyTorch lets you skip the explicit sigmoid by using `BCEWithLogitsLoss`, which is numerically more stable:

```python
# Instead of this:
model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
loss_fn = nn.BCELoss()

# Do this (recommended):
model = nn.Linear(2, 1)                # no sigmoid
loss_fn = nn.BCEWithLogitsLoss()       # applies sigmoid + BCE internally
```

---

## Part 6 Logistic Regression in PyTorch (Spam Detection)

Now we put Parts 3–5 together. Cross-entropy loss tells the computer how good or bad its guess is compared with the true answer (spam or not spam): if the guess is close to the truth, the loss is low; if it's far off, the loss is high and the model needs to learn more. In PyTorch, we create a simple model that predicts probabilities, use cross-entropy to measure errors, and use an optimizer to update the model's parameters until it classifies emails correctly.

A logistic regression model is:

1. A **linear layer** (computes $z = \mathbf{w}^T\mathbf{x} + b$)
2. Followed by a **sigmoid activation** (squashes $z$ into $[0, 1]$ to give a probability)

Then:

- **Binary cross-entropy loss** (`nn.BCELoss`) compares predicted probabilities with the actual labels.
- An **optimizer** such as stochastic gradient descent (SGD) updates the parameters based on gradients.

### 6.1 The model in code

```python
import torch
import torch.nn as nn

# Logistic regression = Linear + Sigmoid
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=1),  # linear layer
    nn.Sigmoid()                                # probability output
)

# Loss function
loss_fn = nn.BCELoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

### 6.2 The training loop

This is the same 5-step loop from Section 2.3. Every iteration does:

1. **Forward pass:** make predictions
2. **Compute loss:** compare predictions to true labels
3. **Zero gradients:** clear old gradients
4. **Backward pass:** compute new gradients
5. **Update parameters:** nudge weights in the downhill direction

```python
for epoch in range(num_epochs):
    # 1. Forward pass: predicted probabilities
    y_pred = model(X)
    
    # 2. Compute loss
    loss = loss_fn(y_pred, y)
    
    # 3. Zero out old gradients
    optimizer.zero_grad()
    
    # 4. Backward pass: compute gradients
    loss.backward()
    
    # 5. Update weights
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### 6.3 Prediction after training

After training, predicted probabilities are converted into class predictions using a threshold (commonly 0.5, see Section 3.6):

```python
with torch.no_grad():
    probabilities = model(X)
    predictions = (probabilities > 0.5).float()   # 1 if spam, 0 if not
```

- Probability > 0.5 → predict spam (1)
- Probability ≤ 0.5 → predict not spam (0)

> **Tip:** for training, the recommended setup is the `BCEWithLogitsLoss` version from Section 5.6 (no sigmoid inside the model).

---

## Part 7 Multi-Class Classification with Softmax

### 7.1 From one score to many

In **binary** classification, we have one output neuron that produces a single score, and we apply a **sigmoid** to squash it into a probability. In **multi-class** classification, we have **one output neuron per class**, each producing its own score. **Softmax** then converts these scores into a probability distribution across all classes.

### 7.2 Softmax in plain words

Imagine a carnival booth where you throw darts at 10 targets labeled 0 through 9. Each dart produces a "score" for every target: some higher, some lower. But these scores are wild numbers like 2.3, −1.5, 8.7. They're not easy to interpret.

Softmax is the booth operator who takes those raw scores and converts them into percentages that add up to 100%. For the scores [2.3, −1.5, 8.7], softmax says: "Target 0: ~0.2%, Target 1: ~0.004%, Target 2: ~99.8%." Now we can say, "The model is 99.8% confident this is a 2."

The word "soft" means it doesn't just pick the winner (that would be hard max, or argmax). Instead, it gives every class a slice of probability, weighted by how high its score was. The winner still gets the biggest slice.

### 7.3 The softmax formula

Given raw scores (called **logits**) $z = [z_1, z_2, \dots, z_K]$:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Three things to notice:

1. **Exponentiate** each score → makes everything positive and amplifies differences.
2. **Sum** all the exponentials → this is the denominator.
3. **Divide** each by the sum → every output is between 0 and 1, and all outputs sum to 1.

That last property is why softmax is used as a probability function.

**Why exponentiate?**

- $e^x$ is always positive → no negative "probabilities."
- It exaggerates differences: if $z_1 = 5$ and $z_2 = 1$, then $e^5 / e^1 \approx 55$, so class 1 gets ~55× the probability of class 2.
- It's smooth and differentiable → perfect for gradient descent.
- It's monotonic → the ranking of the logits is preserved.

### 7.4 Worked example: Cat, Dog, or Bird?

Imagine a neural network that looks at an image and classifies it into one of three classes ($K = 3$). The model outputs these raw scores (logits):

- Cat: 2.0
- Dog: 1.0
- Bird: −1.0

They don't add up to 1, and one value is negative. Softmax fixes this in two steps:

1. **Exponentiate** the scores ($e^{\text{logit}}$): negative numbers become positive, and larger numbers stand out more.
2. **Normalize:** divide each exponent by the total so they add up to 1.

| Class | Logit | $e^{\text{logit}}$ | Probability |
|---|---:|---:|---|
| Cat | 2.0 | 7.39 | 7.39 / 10.48 = **0.705** (70.5%) |
| Dog | 1.0 | 2.72 | 2.72 / 10.48 = **0.260** (26.0%) |
| Bird | −1.0 | 0.37 | 0.37 / 10.48 = **0.035** (3.5%) |
| **Total** | | **10.48** | **1.0** |

- **Sum check:** 0.705 + 0.260 + 0.035 = 1.0
- **Order check:** the logit order Cat (2.0) > Dog (1.0) > Bird (−1.0) is preserved in the probabilities: Cat (70.5%) > Dog (26.0%) > Bird (3.5%).

### 7.5 Multi-class classification step by step in PyTorch

#### Step 1: Linear equations generate a logit for each class

For a network with $K$ classes, each class $k$ has its own weight vector $\mathbf{w}_k$ and bias $b_k$. Given an input vector $\mathbf{x}$ (features from the previous layer), the logit for class $k$ is:

$$z_k = \mathbf{w}_k \cdot \mathbf{x} + b_k$$

Stacking all classes together:

$$\mathbf{z} = W\mathbf{x} + \mathbf{b} \qquad \text{where } W \text{ is } (K \times D), \; \mathbf{z} \text{ is } (K,)$$

In PyTorch, this is simply an `nn.Linear` layer:

```python
import torch
import torch.nn as nn

# 4 input features → 3 classes

linear = nn.Linear(in_features=4, out_features=3)

x = torch.tensor([[1.0, 2.0, 0.5, -1.0]])   # shape: (1, 4)
logits = linear(x)                            # shape: (1, 3)
print(logits)
# tensor([[ 0.42, -1.13,  0.87]], grad_fn=<AddmmBackward>)
```

These raw outputs are called **logits**: unnormalized, unbounded real numbers. Each one measures "how strongly the model favors this class" relative to the others. They are not probabilities yet.

#### Step 2: Softmax converts logits into probabilities

$$\text{softmax}(\mathbf{z})_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$$

**Key properties:**

- Every output is in (0, 1)
- All outputs sum to 1
- Larger logits → larger probabilities
- Differences between logits are preserved as ratios of probabilities (a logit gap of 2 → a probability ratio of $e^2 \approx 7.4$)

```python
probs = torch.softmax(logits, dim=1)
print(probs)
# tensor([[0.36, 0.08, 0.56]])
print(probs.sum(dim=1))   # tensor([1.])  ← sums to 1
```

#### Step 3: Argmax selects the predicted class

The predicted class is the one with the highest logit; equivalently, the highest probability (softmax is monotonic):

```python
predicted_class = torch.argmax(logits, dim=1)
print(predicted_class)   # tensor([2])  ← class index 2 has the largest logit
```

`argmax` returns the **index**, not the value. In this example, logit 0.87 was the largest, so class 2 wins.

### 7.6 Putting it together: a full model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassNet(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)
    
    def forward(self, x):
        logits = self.linear(x)          # (batch, n_classes)
        return logits                    # return logits, NOT softmax

model = MultiClassNet(n_features=4, n_classes=3)
x = torch.randn(5, 4)                    # batch of 5 samples
logits = model(x)                        # (5, 3)

# --- Inference ---
probs = F.softmax(logits, dim=1)         # (5, 3), rows sum to 1
preds = torch.argmax(logits, dim=1)      # (5,), class indices

print("Logits:\n", logits)
print("Probabilities:\n", probs)
print("Predicted classes:", preds)
```

### 7.7 Training: use `CrossEntropyLoss`, not softmax + NLL

PyTorch's `nn.CrossEntropyLoss` combines `log_softmax` + negative log-likelihood in one numerically stable operation. So during training we feed it **raw logits**:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# targets: integer class labels, shape (batch,)
targets = torch.tensor([0, 2, 1, 0, 2])

loss = criterion(logits, targets)   # logits, NOT probs
loss.backward()
optimizer.step()
```

> **Warning:** do not apply softmax before `CrossEntropyLoss`. It will be applied twice and hurt training.

### 7.8 Binary vs. multi-class side by side

| | Binary | Multi-class |
|---|---|---|
| Output layer | `nn.Linear(784, 1)` | `nn.Linear(784, 10)` |
| Activation to get probabilities | Sigmoid | Softmax |
| Loss | `nn.BCEWithLogitsLoss()` (sigmoid inside) | `nn.CrossEntropyLoss()` (softmax inside) |
| Predicted class | Threshold the probability | `argmax` |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Binary logistic regression ---
binary_model = nn.Linear(784, 1)          # one output
logits = binary_model(x)                  # [batch, 1]
loss = nn.BCEWithLogitsLoss()(logits.squeeze(), targets)  # sigmoid inside
probs = torch.sigmoid(logits)

# --- Multi-class softmax classifier ---
multi_model = nn.Linear(784, 10)          # ten outputs (one per class)
logits = multi_model(x)                   # [batch, 10]
loss = nn.CrossEntropyLoss()(logits, targets)   # softmax inside
probs = F.softmax(logits, dim=1)          # [batch, 10]
preds = torch.argmax(probs, dim=1)        # [batch]
```

---

## Stage 2 self-check

Try each question first, then open the answer.

**1. List the five steps of the training loop in order.**

<details>
<summary>Show answer</summary>

Forward → Loss → Reset (`zero_grad`) → Backward → Step.

</details>

**2. Why is MSE a poor loss for a sigmoid classifier?**

<details>
<summary>Show answer</summary>

The sigmoid's derivative σ′(z) = ŷ(1−ŷ) becomes tiny when the output is saturated, so the gradient vanishes exactly when the model is confidently wrong. With cross-entropy, that term cancels.

</details>

**3. Why do we feed raw logits to `CrossEntropyLoss` and `BCEWithLogitsLoss`?**

<details>
<summary>Show answer</summary>

They apply softmax or sigmoid internally, in a numerically stable way. Applying softmax ourselves first would apply it twice and hurt training.

</details>

**4. A model outputs the logits `[2.0, 1.0, −1.0]`. What are the softmax probabilities?**

<details>
<summary>Show answer</summary>

About 0.705, 0.260, and 0.035 (they add up to 1).

</details>

**5. Minimizing binary cross-entropy is the same as doing what?**

<details>
<summary>Show answer</summary>

Maximizing the likelihood of the observed labels (maximum likelihood estimation).

</details>

---

# Stage 3: Neural Networks

> **Parts 8 to 10.** By the end of this stage we can:
> - build dense networks with one or many hidden layers (`nn.Linear`, `nn.ModuleList`)
> - trace forward propagation and the tensor shapes at each layer
> - explain how backpropagation computes gradients, and why they can vanish in deep networks

## Part 8 — Neural Networks with Hidden Layers (MNIST)

So far, every model was a single `nn.Linear` layer, which can only draw a straight decision boundary. Adding **hidden layers with a non-linear activation** turns it into a neural network. This part uses one running example: **MNIST** handwritten digits, with the architecture **784 inputs → 100 hidden neurons → 10 outputs**. Later parts reuse it.

### 8.1 From a single input to a feature vector

Real-world data is rarely a single number. An MNIST image is a **28×28 grid of pixels = 784 features**, so each sample is a vector:

$$\mathbf{x} = [x_1, x_2, \dots, x_{784}]^T \in \mathbb{R}^{784}$$

Before the image enters the network, it is flattened into this vector:

```python
x = x.view(x.size(0), -1)     # (batch, 1, 28, 28) → (batch, 784)
```

### 8.2 Dense (fully connected) layers

A **dense (fully connected) layer** takes every input feature and connects it to every neuron in the next layer. For a hidden layer with $H$ neurons:

$$a_j = \varphi\Big(\sum_i w_{ji}\, x_i + b_j\Big), \qquad j = 1, \dots, H$$

In matrix form:

$$\mathbf{z} = W\mathbf{x} + \mathbf{b}, \qquad \mathbf{a} = \varphi(\mathbf{z})$$

where:

- $W$ is the weight matrix of shape `(H, D)` (the same `outputs × features` layout as in Section 2.4)
- $\mathbf{x}$ is the input vector of dimension $D$
- $\mathbf{b}$ is the bias vector of dimension $H$
- $\varphi$ is the activation function (ReLU, tanh, Swish, …)

**Why "dense"?** Every input connects to every neuron, so the weight matrix is fully populated. This contrasts with convolutional layers, where connections are local.

#### PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, in_features=784, hidden=100, out_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)   # W1: (100, 784), b1: (100,)
        self.fc2 = nn.Linear(hidden, out_classes)   # W2: (10, 100),  b2: (10,)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)     # flatten 28×28 → 784
        z1 = self.fc1(x)              # linear transform
        a1 = F.relu(z1)               # hidden activation
        z2 = self.fc2(a1)             # output logits
        return z2
```

Shapes and parameter counts for this network:

| Layer | Weight | Bias | Output | Parameters |
|---|---|---|---|---:|
| `fc1` (hidden) | `(100, 784)` | `(100,)` | `(100,)` | 78,500 |
| `fc2` (output) | `(10, 100)` | `(10,)` | `(10,)` | 1,010 |
| **Total** | | | | **79,510** |

### 8.3 Hidden neurons: what do they learn?

Each hidden neuron computes a **weighted sum of all inputs**, then applies a non-linearity. Stacking many such neurons lets the network learn **increasingly abstract features**:

- **Early layers:** edges, strokes, pixel intensities
- **Middle layers:** digit parts (loops, curves, crossings)
- **Output layer:** class-discriminative combinations

> **Note:** this early → middle → output progression is the general pattern in deeper networks. The MNIST model above has a single hidden layer, so there it is only a rough picture.

Without a non-linear activation like ReLU, stacked dense layers would collapse into a single linear map, no matter how many layers we add.

### 8.4 Forward propagation

**Forward propagation** is the process of pushing an input through the network, layer by layer, to produce an output. Each layer applies a linear transformation followed by an activation.

#### Step by step for MNIST

Given input $\mathbf{x}$ (flattened to 784):

$$
\begin{aligned}
\mathbf{z}^1 &= W^1\mathbf{x} + \mathbf{b}^1 && \text{linear transform into the hidden layer} \\
\mathbf{a}^1 &= \varphi(\mathbf{z}^1) && \text{non-linear activation (ReLU)} \\
\mathbf{z}^2 &= W^2\mathbf{a}^1 + \mathbf{b}^2 && \text{linear transform into the output layer} \\
\hat{\mathbf{y}} &= \text{softmax}(\mathbf{z}^2) && \text{probabilities over 10 classes (handled by the loss)}
\end{aligned}
$$

Where:

- $\mathbf{z}^1$ has shape `(100,)`: the pre-activation of the hidden layer
- $\mathbf{a}^1$ has shape `(100,)`: the hidden activations
- $\mathbf{z}^2$ has shape `(10,)`: the **logits** for the 10 digit classes
- $\hat{\mathbf{y}}$ has shape `(10,)`: the predicted probability distribution

#### The output layer (multiclass)

The output layer has **one neuron per class**. Its computation is a linear transformation of the hidden activations:

$$\mathbf{z} = W\mathbf{a} + \mathbf{b}$$

where $\mathbf{a}$ is the hidden activation, $W$ is the weight matrix (shape `(10, 100)`), and $\mathbf{b}$ is the bias. The predicted class is the neuron with the **largest output value**:

```python
predicted_class = torch.argmax(logits, dim=1)
```

#### Full forward pass on a batch

```python
model = DenseNet()
x = torch.randn(64, 1, 28, 28)          # batch of 64 MNIST images
logits = model(x)                        # shape: (64, 10)
probs  = F.softmax(logits, dim=1)        # shape: (64, 10), sums to 1
preds  = torch.argmax(logits, dim=1)     # shape: (64,)
```

### 8.5 Multiclass network design

| Property | Value |
|---|---|
| Neurons in output layer | **One per class** (10 for MNIST) |
| Activation | **None** (raw logits) |
| Loss function | `nn.CrossEntropyLoss` |
| Prediction rule | `argmax` over logits |

**Why drop the sigmoid/softmax activation at the output?** As in Section 7.7, `nn.CrossEntropyLoss` applies `log_softmax` internally and computes the negative log-likelihood in one numerically stable operation. So during training we do **not** apply softmax to the output layer; we feed **raw logits** directly:

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)   # logits, NOT probs
```

#### MNIST architecture reference

```text
Input:  784 features (28×28 flattened)
Hidden: 100 neurons, ReLU activation
Output: 10 neurons (one per digit 0–9), no activation
Loss:   nn.CrossEntropyLoss (softmax + NLL built in)
```

### 8.6 Training loop with validation

This is the same 5-step loop from Section 2.3, now with a validation pass after every epoch:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)     # softmax applied internally
        loss.backward()                 # backprop
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for x, y in val_loader:
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        print(f"Epoch {epoch}: val_acc = {correct/total:.4f}")
```

**Training and evaluation pattern:**

- Track **training loss per iteration**: it confirms the model is fitting.
- Track **validation accuracy per epoch**: it confirms the model generalizes.
- `nn.CrossEntropyLoss` handles softmax internally, so we feed logits, not probabilities.

> **How the pieces connect:** the loop calls `loss.backward()` as a black box. Part 9 adds more hidden layers, and Part 10 opens the box and explains what `backward()` computes and why it can fail in deep networks. The `Adam` optimizer used here is explained in Part 12.

---

## Part 9 — Deep Neural Networks

Part 8 used a single hidden layer. This part adds **more hidden layers** and shows how to build them without naming every layer by hand.

### 9.1 What makes a network "deep"?

Adding **more hidden layers** (not just more neurons in one layer) creates a **deep neural network**. Depth lets the model learn **hierarchical features**: early layers capture simple patterns, and deeper layers combine them into complex abstractions (the same early → middle → output progression described in Section 8.3).

**Key advantages of depth over width:**

- Often improves performance, and can reduce overfitting, compared with just widening a single hidden layer
- Hidden layers can have **different numbers of neurons**, enabling more complex decision functions
- More parameter-efficient: depth reuses features across layers

> **Note:** deeper models can also overfit, and they are harder to train (Part 10). Parts 11 to 13 cover the tools that make deep networks work in practice.

**Architecture pattern:**

```text
Input (D_in) → Hidden₁ (H1) → Hidden₂ (H2) → ... → Output (D_out)
```

- `D_in` = input dimension (e.g., 784 for MNIST)
- `H1, H2, ...` = neurons per hidden layer (can vary)
- `D_out` = number of classes (e.g., 10 for digits 0–9)

**Forward pass:** each hidden layer applies a linear transform followed by a non-linear activation (ReLU, tanh, sigmoid). The output layer stays linear and returns raw logits, exactly as in Section 8.5.

**MNIST example:** 784 → 50 → 50 → 10, trained with SGD and cross-entropy. **ReLU and tanh outperform sigmoid** in both loss and validation accuracy. (Sigmoid saturates, which is the vanishing-gradient effect from Section 10.4.)

### 9.2 Building deep networks with `nn.ModuleList()`

When the number of layers is dynamic, manually naming `fc1, fc2, fc3, ...` is impractical. `nn.ModuleList()` lets us define layers **programmatically in a loop**.

**Construction pattern:**

```python
import torch.nn as nn
import torch.nn.functional as F

class DeepNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])
    
    def forward(self, x):
        for layer in self.layers[:-1]:       # all but last
            x = F.relu(layer(x))             # activation
        return self.layers[-1](x)            # last: linear only

model = DeepNet([784, 100, 50, 10])
```

**Key design points:**

- `layer_sizes = [784, 100, 50, 10]` → 3 linear layers
- **All but the last layer** apply ReLU
- **The last layer applies linear only**: raw logits for multiclass classification with `CrossEntropyLoss`

> **Why `ModuleList` and not a plain Python list?** PyTorch only registers the parameters of layers held in a `ModuleList` (or assigned as attributes). Layers kept in a plain list would be invisible to `model.parameters()`, so the optimizer would never update them.

Shapes and parameter counts for `[784, 100, 50, 10]` (compare with the one-hidden-layer table in Section 8.2):

| Layer | Weight | Bias | Parameters |
|---|---|---|---:|
| `layers[0]` (784 → 100) | `(100, 784)` | `(100,)` | 78,500 |
| `layers[1]` (100 → 50) | `(50, 100)` | `(50,)` | 5,050 |
| `layers[2]` (50 → 10) | `(10, 50)` | `(10,)` | 510 |
| **Total** | | | **84,060** |

**Experimentation:** easily test different depths and widths by changing `layer_sizes`.

> **How the pieces connect:** every extra layer adds another derivative to the product computed by backpropagation. Part 10 shows why that product can shrink toward zero in deep networks, and Parts 11 to 13 fix it.

---

## Part 10 — Backpropagation and the Vanishing Gradient Problem

### 10.1 Backpropagation basics

**Backpropagation** is the algorithm used to compute the gradients of the loss with respect to every parameter in the network. It relies on the **chain rule** of calculus:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w}$$

By breaking the derivative down layer by layer, backprop avoids recomputing shared terms. It **reuses intermediate gradient terms** from the output layer to compute gradients in earlier layers. This is why it is efficient: one forward pass plus one backward pass per update.

### 10.2 Worked example: one hidden layer and one output layer

Consider a tiny network with input $\mathbf{x}$, hidden activation $\mathbf{a} = \varphi(W^1\mathbf{x} + \mathbf{b}^1)$, and output $\hat{\mathbf{y}} = W^2\mathbf{a} + \mathbf{b}^2$, with loss $L(\hat{\mathbf{y}}, \mathbf{y})$.

**Step 1: gradient of the loss with respect to the output-layer parameters**

$$\frac{\partial L}{\partial W^2} = \frac{\partial L}{\partial \hat{\mathbf{y}}}\, \mathbf{a}^T, \qquad \frac{\partial L}{\partial \mathbf{b}^2} = \frac{\partial L}{\partial \hat{\mathbf{y}}}$$

**Step 2: propagate the gradient back to the hidden layer**

$$\boldsymbol{\delta}^2 = \frac{\partial L}{\partial \hat{\mathbf{y}}} \quad \text{(gradient at the output)}$$

$$\boldsymbol{\delta}^1 = \big((W^2)^T \boldsymbol{\delta}^2\big) \odot \varphi'(\mathbf{z}^1) \quad \text{(gradient at the hidden layer)}$$

**Step 3: gradients of the hidden-layer parameters**

$$\frac{\partial L}{\partial W^1} = \boldsymbol{\delta}^1\, \mathbf{x}^T, \qquad \frac{\partial L}{\partial \mathbf{b}^1} = \boldsymbol{\delta}^1$$

The gradient from the output layer is **reused** to compute the hidden layer's gradient. This is the core efficiency of backprop.

| Direction | What is computed | Result |
|---|---|---|
| **Forward** | $\mathbf{z}^1 \to \mathbf{a}^1 \to \mathbf{z}^2 \to \hat{\mathbf{y}} \to L$ | The prediction and the loss |
| **Backward** | $\boldsymbol{\delta}^2 \to \partial L/\partial W^2,\ \partial L/\partial \mathbf{b}^2 \to \boldsymbol{\delta}^1 \to \partial L/\partial W^1,\ \partial L/\partial \mathbf{b}^1$ | Gradients for every parameter |

### 10.3 PyTorch handles this automatically

```python
loss.backward()      # autograd computes all gradients via chain rule
optimizer.step()     # updates parameters using those gradients
```

PyTorch's **autograd** engine (introduced in Section 1.3) builds a computational graph during the forward pass and traverses it in reverse during `backward()`.

### 10.4 The vanishing gradient problem

In **deep networks**, gradients are products of many derivatives, one per layer:

$$\frac{\partial L}{\partial W^1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a^N} \cdot \frac{\partial a^N}{\partial a^{N-1}} \cdots \frac{\partial a^2}{\partial a^1} \cdot \frac{\partial a^1}{\partial z^1} \cdot \frac{\partial z^1}{\partial W^1}$$

If each layer's derivative is **less than 1** (a sigmoid's derivative never exceeds 0.25, and a saturated tanh's is close to 0), the product **shrinks exponentially** as we go deeper. Early layers receive extremely tiny gradients and barely update, so the network stops learning meaningful features.

> **Related idea:** Section 5.2 showed a one-layer version of the same effect: a saturated sigmoid output with MSE gives a tiny gradient when the model is confidently wrong.

#### Symptoms

- Early-layer weights barely change during training
- Training loss plateaus early
- Accuracy stalls regardless of more epochs or data

### 10.5 Solutions (preview)

| # | Solution | Idea |
|---|---|---|
| 1 | **Change activation functions** | Use ReLU or Swish instead of sigmoid or tanh; they don't saturate on the positive side |
| 2 | **Use advanced optimizers** | Momentum, Adam, and RMSProp keep gradients moving and adapt learning rates per parameter (Part 12) |
| 3 | **Initialization and architectural fixes** | Careful weight initialization (He/Xavier) and batch normalization (Part 11); residual connections (ResNet, Part 15) |
| 4 | **Gradient clipping** | Caps *exploding* gradients, the opposite problem, mainly in RNNs |

> **How the pieces connect:** the activation choice (ReLU in `DenseNet`) already helps. Part 11 covers initialization and batch normalization, Part 12 the optimizer side, and Part 15 residual connections.

---

## Stage 3 self-check

Try each question first, then open the answer.

**1. For the MNIST network 784 → 100 → 10, what is the shape of the first weight matrix, and how many parameters does the network have?**

<details>
<summary>Show answer</summary>

`(100, 784)`, and 79,510 parameters in total (78,500 in the first layer and 1,010 in the second).

</details>

**2. Why does a hidden layer need a non-linear activation?**

<details>
<summary>Show answer</summary>

Without it, stacked dense layers collapse into a single linear map, no matter how many layers we add.

</details>

**3. What does `nn.ModuleList` do that a plain Python list of layers does not?**

<details>
<summary>Show answer</summary>

It registers the layers' parameters with the model, so `model.parameters()` (and therefore the optimizer) can see them.

</details>

**4. Why can gradients vanish in a deep network?**

<details>
<summary>Show answer</summary>

The gradient is a product of one derivative per layer. If each is below 1 (a sigmoid's derivative never exceeds 0.25), the product shrinks exponentially toward the early layers.

</details>

**5. What does backpropagation reuse to be efficient?**

<details>
<summary>Show answer</summary>

The gradient at the output layer (δ²) is reused to compute the hidden-layer gradient (δ¹), so one forward and one backward pass are enough per update.

</details>

---

# Stage 4: Training Deep Networks Well

> **Parts 11 to 13.** By the end of this stage we can:
> - choose a weight initialization to match the activation function
> - use batch normalization, momentum, adaptive optimizers, and learning-rate schedules
> - reduce overfitting with dropout, and switch correctly between training and evaluation modes

## Part 11 — Weight Initialization and Batch Normalization

Part 10 ended with a list of fixes for vanishing gradients. This part covers two of them: **how the weights start** (initialization) and **keeping activations well-scaled during training** (batch normalization). Momentum, the third helper, is in Part 12.

### 11.1 Why initialization matters

#### The symmetry problem

If we initialize **all weights to the same value**, every neuron in a layer produces **identical outputs and identical gradients**. They stay identical forever, so the network can never learn **distinct features**.

**Fix:** break symmetry with **random initialization**.

#### The range problem

Random uniform initialization requires a **carefully chosen range**:

| Range | Effect |
|---|---|
| **Too narrow** | Weights cluster near zero → small variance → slow learning |
| **Too wide** | Large outputs push activations into saturated regions → **vanishing gradients** |
| **Just right** | Activations stay where derivatives are meaningful → gradients flow |

#### Why large outputs kill learning

Large linear outputs (e.g., into `tanh`) land in the **flat tails** of the activation, where the derivative is near zero. Gradients shrink, and early layers stop updating.

#### The scaling insight

**Scale the initialization range by $1/\sqrt{L_{in}}$** (the inverse square root of the number of input neurons). This keeps the weighted sum's variance stable regardless of layer width, so activations stay in a healthy range.

> More input neurons → bigger weighted sums → more aggressive scaling needed.

#### Recap: why initialization fails

1. **Symmetry:** identical weights → identical neurons → no learning
2. **Bad range:** too narrow (slow) or too wide (vanishing gradients)
3. **Layer size:** wider layers amplify the weighted sum, worsening vanishing gradients unless the weights are scaled

> **The core rule:** adjust the initialization width based on the number of input neurons, to keep the initial activations within the functional range of the activation function.

### 11.2 Xavier and He initialization

#### PyTorch's default initialization

PyTorch initializes `nn.Linear` weights **uniformly** in:

$$\left[-\frac{1}{\sqrt{L_{in}}},\ +\frac{1}{\sqrt{L_{in}}}\right]$$

This default scaling already helps keep gradients stable.

#### Xavier (Glorot) initialization: for `tanh`

Xavier considers **both input and output** neuron counts:

$$\text{range} = \left[-\sqrt{\frac{6}{L_{in} + L_{out}}},\ +\sqrt{\frac{6}{L_{in} + L_{out}}}\right]$$

**Why:** it balances variance across layers, keeping activations in `tanh`'s responsive region.

```python
nn.init.xavier_uniform_(layer.weight)
```

**Result:** faster convergence and better early-training accuracy for `tanh` networks.

#### He (Kaiming) initialization: for `ReLU`

He initialization accounts for the fact that **ReLU zeroes out half the neurons** (negative inputs → 0), effectively halving the active signal. It compensates by using a wider range (for the uniform version, $\pm\sqrt{6/L_{in}}$).

```python
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
```

**Result:** better accuracy for ReLU-based networks.

#### Quick chooser

| Activation | Initialization |
|---|---|
| `tanh`, `sigmoid` | **Xavier** |
| `ReLU`, `Leaky ReLU` | **He (Kaiming)** |
| Default / unsure | PyTorch default ($1/\sqrt{L_{in}}$) |

### 11.3 Batch normalization

#### What it does

Batch normalization **normalizes neuron outputs before the activation function**, using the **mean and standard deviation of the current mini-batch**. Then it **scales and shifts** the result with learnable parameters, giving the model flexibility.

#### Step by step (per mini-batch, per neuron)

```text
1. Compute mean μ and std σ from the mini-batch
2. Normalize:      x̂ = (x - μ) / σ
3. Scale & shift:  y = γ·x̂ + β     (γ, β are learnable)
4. Pass through the activation
```

> In practice, a tiny constant ε is added inside the square root of the variance to avoid dividing by zero.

#### Training vs. prediction

| Phase | Statistics used |
|---|---|
| **Training** | Current **mini-batch** mean/std |
| **Prediction** | **Population** mean/std (running averages) |

#### In PyTorch

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.bn1 = nn.BatchNorm1d(100)      # specify # neurons
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = self.bn1(self.fc1(x))           # normalize before activation
        x = F.relu(x)
        return self.fc2(x)
```

**Critical:** use `model.train()` during training and `model.eval()` during prediction, because batch norm behaves differently in each mode (the table above). Dropout has the same requirement (Part 13).

#### Benefits

- Stabilizes and accelerates training
- Reduces sensitivity to initialization
- Acts as a mild regularizer
- Often allows higher learning rates

> **How the pieces connect:** initialization keeps gradients flowing from the very first step, and batch normalization keeps activations well-scaled throughout training. Part 12 adds momentum and adaptive optimizers on top.

---

## Part 12 — Better Training: Momentum, Optimizers, and Learning-Rate Strategies

So far we have mostly used plain SGD (the Part 8 training loop already used Adam). This part starts with momentum, then explains smarter optimizers and ways to adjust the learning rate during training. Together they are the optimizer side of the vanishing-gradient solutions in Section 10.5.

### 12.1 The mountain-in-the-fog analogy

Imagine you're trying to find the fastest way down a mountain in the fog. The mountain is your model's **error**, and you want to reach the bottom (lowest error) as quickly and safely as possible. The **optimizer** is your guide, deciding which path to take and how big your steps should be.

- **Adam** is a smart guide who watches how steep the path is and adjusts your step size **for each foot separately**. If one foot is on a slippery slope, it takes smaller steps; if the other is on a gentle slope, it takes bigger steps.
- **RMSProp** looks at how steep the path has been **recently** and adjusts your step size to avoid big jumps — great for uneven terrain.
- **AdamW** is like Adam but better at keeping your shoes clean (**regularization**). It separates the cleaning from the walking, so your shoes last longer.

Keep this analogy in mind — everything below builds on it.

### 12.2 Momentum

Momentum accelerates gradient descent by **accumulating past gradients** into a velocity term, smoothing the updates and helping escape flat regions. Think of a ball rolling downhill: it builds up speed in a consistent direction. This is especially useful when gradients are small because of poor initialization (Section 11.1) or saturation (Section 10.4).

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Adam (next section) keeps a momentum-style running average of the gradients, which is its "first moment".

### 12.3 Optimizers: Adam, RMSProp, and AdamW

#### Adam (Adaptive Moment Estimation)

Adam adapts the learning rate **individually for each parameter** using running averages of gradients.

- Keeps a **first moment** (mean of gradients) and a **second moment** (mean of squared gradients).
- Divides the update by the square root of the second moment — so noisy parameters get smaller steps.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

- **Pros:** fast convergence, robust to learning rate choice.
- **Cons:** weight decay is coupled with gradient updates, which can hurt generalization.

#### RMSProp

RMSProp normalizes updates by maintaining a moving average of squared gradients.

- Doesn't use momentum like Adam, just a scaled gradient.
- Excellent for sequential models (RNNs, LSTMs) where gradients vary wildly.

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
```

- **Pros:** stable in non-stationary settings.
- **Cons:** no momentum, can be slower than Adam on some tasks.

#### AdamW (Adam with decoupled weight decay)

AdamW is like Adam, but weight decay is applied separately from the gradient update.

- In classic Adam + L2, decay gets scaled by the adaptive learning rate, which weakens its effect.
- AdamW decouples them, giving better regularization and generalization, especially in large networks.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

- **Pros:** better generalization, current default for transformers.
- **Cons:** slightly more hyperparameters to tune.

#### Comparison

| Optimizer | Key idea | Best for | Main trade-off |
|---|---|---|---|
| **Adam** | Per-parameter step sizes from running averages of gradients | General use; fast convergence | Weight decay coupled with updates |
| **RMSProp** | Scaled gradient from a moving average of squared gradients, no momentum | RNNs / LSTMs with wildly varying gradients | Can be slower than Adam |
| **AdamW** | Adam with weight decay applied separately | Large networks, transformers | Slightly more hyperparameters |

### 12.4 Learning-rate strategies

| Strategy | Idea | PyTorch class |
|---|---|---|
| **Step decay** | Reduce the LR by a factor every N epochs | `StepLR` |
| **Plateau reduction** | Reduce when a monitored metric (e.g., validation loss) stops improving | `ReduceLROnPlateau` |
| **Cosine / gradual decay** | Smoothly decay the LR along a cosine curve; very popular in modern training | `CosineAnnealingLR` |
| **Warm-up** | Start with a very small LR, then gradually increase; stabilizes early training, especially in large models | `LambdaLR` |

#### Step decay

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

#### Plateau reduction

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)
# In training loop:
scheduler.step(val_loss)
```

#### Cosine / gradual decay

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

#### Warm-up

```python
def warmup_lambda(epoch):
    if epoch < 5:
        return (epoch + 1) / 5
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
```

---

## Part 13 — Overfitting and Dropout

### 13.1 The problem: memorizing the training data

Deep networks with many parameters can **memorize** the training data, which leads to poor generalization. The warning sign is the one tracked in Section 8.6: training loss keeps improving while **validation accuracy** stalls or drops.

**Dropout's solution:** randomly **deactivate neurons** during training, so the network can't rely on any single path.

### 13.2 How dropout works

- In PyTorch, `nn.Dropout(p)` turns each neuron **off with probability `p`** and keeps it with probability `1 - p`
- A **Bernoulli random variable** (the same distribution as the coin flip in Section 4.1) acts as a mask: `0` (off) or `1` (on)
- The masks are applied **independently** across layers and iterations

> **Note:** `p` is the probability of *dropping* a neuron, not of keeping it. This matches the scaling factor and the recommended values below.

**Training vs. evaluation:**

| Phase | Dropout |
|---|---|
| **Training** | ON: random neurons disabled |
| **Evaluation / inference** | OFF: full network used |

**PyTorch uses *inverted dropout*:** during training, the surviving activations are scaled by $1/(1-p)$, so the **expected value stays consistent** between the train and eval phases. No scaling is needed at inference.

**Choosing `p`:**

| Layer size | Suggested `p` |
|---|---|
| Larger layers | Higher `p` (up to **0.5**) |
| Smaller layers | Lower `p` (0.05–0.1) |

### 13.3 Using dropout in PyTorch

**Effect:** dropout forces the model to learn **more robust features**, reducing overlap in decision boundaries. In the course example, it improved validation accuracy from **77% → 87%**.

**Implementation:**

```python
class Net(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.drop = nn.Dropout(p)            # dropout layer
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)                      # apply dropout
        return self.fc2(x)
```

**Placement:** dropout layers go **between hidden layers** (not on the output layer).

**Critical: toggling modes**

```python
model.train()   # enables dropout
model.eval()    # disables dropout (for validation/inference)
```

These are the same two calls already used in the training loop of Section 8.6. They also switch batch normalization (Section 11.3) between its training and prediction behavior.

**Typical training setup:** Adam optimizer, cross-entropy loss, batch gradient descent, tracking metrics per epoch.

### 13.4 Dropout as regularization and generalization

**Dropout is a regularization technique** that reduces overfitting by randomly disabling neurons during training.

**Two powerful interpretations:**

1. **Bernoulli masking:** a stochastic mask applied to the activations at each step
2. **Ensemble of smaller networks:** each training step samples a different sub-network. At inference, the full network approximates an **average of many sub-networks**

**During inference:** dropout is **disabled** for stable, deterministic predictions.

**Key takeaways:**

- Improves **generalization** by preventing co-adaptation of neurons
- Effectiveness depends on **model size, architecture, and training conditions**
- Not a universal fix: combine it with other techniques

#### The regularization toolbox

| Technique | Idea | Where in this guide |
|---|---|---|
| **Dropout** | Randomly disable neurons during training | `nn.Dropout(p)` (this part) |
| **Batch normalization** | Mild regularizing side effect | `nn.BatchNorm1d` (Section 11.3) |
| **Weight decay** | Penalize large weights | `AdamW(weight_decay=...)` (Section 12.3) |
| **Early stopping** | Stop when validation performance stops improving | Validation tracking (Section 8.6) |
| **Data augmentation** | Create more varied training examples | Not covered in this guide |

> **How the pieces connect:** depth (Part 9) gives the model power, and dropout keeps that power from turning into overfitting. Part 14 applies the same ideas to images.

---

## Stage 4 self-check

Try each question first, then open the answer.

**1. Why must we not initialize all weights to the same value?**

<details>
<summary>Show answer</summary>

Every neuron in a layer would produce identical outputs and identical gradients, and stay identical forever, so the layer could never learn distinct features.

</details>

**2. Which initialization suits `ReLU`, and which suits `tanh`?**

<details>
<summary>Show answer</summary>

He (Kaiming) for ReLU, Xavier (Glorot) for tanh and sigmoid. PyTorch's default is already scaled by 1/√L_in.

</details>

**3. Which statistics does batch normalization use during training and during prediction?**

<details>
<summary>Show answer</summary>

Training: the mean and standard deviation of the current mini-batch. Prediction: the population (running-average) mean and standard deviation.

</details>

**4. What does `nn.Dropout(0.3)` do?**

<details>
<summary>Show answer</summary>

During training it turns each neuron off with probability 0.3 and scales the survivors by 1/(1−p) (inverted dropout). In evaluation mode it does nothing.

</details>

**5. Why call `model.eval()` before validation or inference?**

<details>
<summary>Show answer</summary>

It switches dropout off and makes batch normalization use its running statistics. Both behave differently from training mode.

</details>

**6. What does momentum add to plain SGD?**

<details>
<summary>Show answer</summary>

It accumulates past gradients into a velocity term, which smooths the updates and helps move through flat regions.

</details>

---

# Stage 5: Images and Modern Practice

> **Parts 14 to 16.** By the end of this stage we can:
> - build a CNN and work out its layer shapes
> - explain residual connections, and adapt a pretrained model through transfer learning
> - train on a GPU, and know the options for reducing boilerplate and deploying a model

## Part 14 — Convolutional Neural Networks (CNNs)

So far, images were flattened into vectors (Section 8.1) and fed to dense layers. This part builds from the **motivation for convolution** to a complete CNN, using the same MNIST digits.

### 14.1 Why convolution?

**The problem:** a simple vector representation treats each pixel position as independent. If an image shifts by a few pixels, the entire input vector changes, and the model must relearn the pattern at every position.

**The solution:** convolution focuses on **relative pixel positions** rather than absolute ones. A **kernel** (a small learnable filter) slides across the image, detecting the same feature wherever it appears.

### 14.2 The convolution operation

Convolution is essentially a **linear transformation applied locally**:

$$\text{activation\_map}[i,j] = \sum \big(\text{kernel} \odot \text{image\_region}\big) + \text{bias}$$

- Each element of the activation map is the **dot product** of the kernel and a local image region
- A **bias** is added uniformly across the map

The kernel slides **horizontally and vertically** across the image to produce the full activation map.

#### Key hyperparameters

| Parameter | Role |
|---|---|
| **Kernel size** | Size of the sliding filter (e.g., 3×3, 5×5) |
| **Stride** | How far the kernel moves each step |
| **Padding** | Zero-padding around the image border |

**Output size formula:**

$$H_{out} = \left\lfloor \frac{H_{in} + 2\cdot\text{padding} - \text{kernel\_size}}{\text{stride}} \right\rfloor + 1$$

For example, a 16×16 image with `kernel_size=5`, `padding=2`, `stride=1` gives $(16 + 4 - 5)/1 + 1 = 16$: the size is preserved.

### 14.3 Activation functions and max pooling

#### ReLU activation

ReLU is applied **element-wise** to each activation map, preserving the tensor shape:

$$\text{ReLU}(x) = \max(0, x)$$

- Negative values → 0
- Positive values → unchanged
- Applied across **all channels** to introduce non-linearity

#### Max pooling

**Purpose:** downsample the activation maps by selecting the **maximum value** in each region.

**Benefits:**

- Reduces spatial dimensions → fewer parameters and less computation
- Provides **translation invariance**: small shifts in the input don't change the output
- Focuses on **dominant features**

#### PyTorch implementation

```python
import torch.nn as nn

# Convolution + ReLU grouped together
block = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)     # default stride = kernel size
)
```

`nn.Sequential` streamlines building repeated conv → activation → pool blocks.

### 14.4 Multiple input and output channels

#### Multiple output channels

Each output channel has its **own kernel and bias**, producing a separate activation map. Different kernels detect different features (edges, lines, textures).

```text
Output channels = number of kernels
```

#### Multiple input channels (e.g., RGB)

For an image with `C` input channels, each channel is convolved with **its own kernel**, and the results are **summed** to produce one output map (a convolution-based dot product).

#### Multiple input and output channels

When both are multiple:

- Each **output channel** has a **set of kernels**, one per input channel
- Convolution is performed per (input, output) channel pair, then **summed**

**Matrix analogy:**

- Kernels → matrix elements
- Input channels → columns
- Output channels → rows

```python
# 3 input channels (RGB), 32 output channels
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
# Weight shape: (32, 3, 3, 3)  →  32 kernels, each (3, 3, 3)
```

### 14.5 CNN structure

#### Architecture components

```text
Input → [Conv → ReLU → Pool] × N → Flatten → Fully Connected → Output
```

| Component | Role |
|---|---|
| **Convolutional layers** | Extract spatial features |
| **Activation functions** | Introduce non-linearity |
| **Pooling layers** | Reduce spatial dimensions |
| **Fully connected layer** | Classify based on the extracted features |

The last stage is the same multiclass output design as Section 8.5: one neuron per class, raw logits, `CrossEntropyLoss`.

#### Toy example: horizontal vs. vertical lines

A simple CNN with two convolutional layers plus an output layer can distinguish horizontal from vertical lines. It shows how kernels learn orientation-selective features.

#### PyTorch CNN skeleton

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Linear(32 * 4 * 4, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)       # flatten
        return self.fc(x)                # class scores
```

> **Note:** the `32 * 4 * 4` in the output layer is tied to **16×16 input images** (see the shape table in Section 14.6). For 28×28 images, the final feature map would be 32×7×7, so the size would need to change.

#### Training

This is the standard training loop from Section 8.6: define the datasets, model, loss (`CrossEntropyLoss`), optimizer, and train/validation loaders. **Backpropagation updates both the convolutional and the fully connected parameters.**

### 14.6 CNN on MNIST (16×16)

Smaller images (16×16 instead of 28×28) speed up experimentation while keeping the architecture representative.

| Layer | Details | Output shape |
|---|---|---|
| Conv1 | 1→16 channels, kernel 5, padding 2 | (16, 16, 16) |
| MaxPool | kernel 2 | (16, 8, 8) |
| Conv2 | 16→32 channels, kernel 5, padding 2 | (32, 8, 8) |
| MaxPool | kernel 2 | (32, 4, 4) |
| Flatten | 32 × 4 × 4 | 512 |
| FC | 512 → 10 | 10 |

**Forward pass:**

```text
Input → Conv1 → ReLU → Pool → Conv2 → ReLU → Pool → Flatten → FC → Class scores
```

The final feature map (32 channels × 4×4 = **512 elements**) is flattened and fed into a **10-neuron linear layer**, one per digit class.

> **How the pieces connect:** a CNN replaces the first dense layers of Part 8 with conv blocks that respect the image's spatial structure. Stacking many such blocks brings back the vanishing-gradient problem from Part 10, which Part 15 addresses.

---

## Part 15 — ResNets and Transfer Learning

### 15.1 Problems in deep networks

1. **Vanishing gradients:** gradients shrink exponentially, slowing learning in the early layers (Part 10)
2. **Degradation:** adding more layers can *worsen* performance, because deeper networks struggle to learn identity mappings

### 15.2 Residual networks (ResNets)

ResNets introduce **skip connections** that let information bypass layers:

$$\text{output} = F(x) + x$$

- $F(x)$ is the **residual function** learned by the block
- $x$ is the original input (the identity shortcut)
- Learning $F(x) \approx 0$ is easy, so the block can learn **near-identity mappings**

**Benefits:**

- **Improved gradient flow** during backprop: gradients have a direct path to the early layers
- **Stabilized training:** enables networks with **hundreds or thousands of layers**
- Widely used in computer vision (image classification, detection, segmentation)

```text
Input ──┬──→ [Conv → BN → ReLU → Conv → BN] ──→ (+) ──→ ReLU ──→ Output
        └────────────── skip connection ──────────┘
```

### 15.3 Transfer learning and pretrained models

**Transfer learning** means reusing a model trained on a large dataset (e.g., ImageNet) for a related task:

- **Reuse** the pretrained model
- **Freeze the backbone** to retain the learned features
- **Fine-tune** by unfreezing some or all layers to adapt to the new task
- Benefits: **faster training, less data needed, better performance**

| Strategy | What is trained |
|---|---|
| **Frozen backbone** | Only a new output layer |
| **Fine-tuning** | The new output layer plus some or all unfrozen backbone layers |

#### Using a pretrained model (e.g., ResNet18 from TorchVision)

Pretrained models come with **expertly optimized parameters** from training on large datasets (ImageNet).

**Requirements:**

- Inputs must be **color images** (3 channels)
- Images must be **normalized** with specific channel-wise mean and standard deviation (for ImageNet models: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

#### Adapting the model for our data

```python
from torchvision import models
import torch.nn as nn

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# 1. Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# 2. Replace the output layer for our # classes
model.fc = nn.Linear(model.fc.in_features, 7)   # e.g., 7 classes
```

Only the **new output layer** is trained; the backbone is frozen.

> **Note:** older tutorials write `models.resnet18(pretrained=True)`. That argument has been deprecated since torchvision 0.13, so the `weights=` form above is the current one.

#### Training setup

- Dataset objects for train/test with proper transforms
- Data loaders with appropriate batch sizes
- **Cross-entropy loss**
- An optimizer updating **only the new output layer's parameters**
- Switch between `model.train()` and `model.eval()` each epoch

```python
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

**Why this works:** the pretrained backbone already extracts rich features, so we only need to learn how to map them to our classes.

---

## Part 16 — Training at Scale and Deployment

### 16.1 GPU acceleration with PyTorch

#### CUDA and `torch.cuda`

**CUDA** is NVIDIA's parallel computing platform for GPU acceleration. PyTorch's `torch.cuda` package provides:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())        # True/False
print(torch.cuda.get_device_name(0))    # e.g., "NVIDIA RTX 3090"
```

#### Moving tensors and models to the GPU

```python
# Move a tensor
x = x.to(device)

# Move the model (converts all layers to CUDA tensors)
model = model.to(device)
```

#### Training and testing on the GPU

**During training:** send **both features and labels** to the GPU:

```python
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
```

**During testing:** send **only the inputs**, because labels aren't needed for the loss or optimization:

```python
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)                # labels stay on CPU
        logits = model(x)
        preds = logits.argmax(dim=1)
```

> **Note:** to compare predictions with labels (e.g., to compute accuracy), both must be on the same device, for example `(preds.cpu() == y).sum()`.

**Why GPUs matter:** matrix multiplications (the core of convolution and dense layers) are highly parallelizable, so GPUs can be **10–100× faster** than CPUs for CNN training.

### 16.2 PyTorch Lightning

- Abstracts repetitive boilerplate (training loops, logging, checkpointing)
- Built-in support for **distributed training**, **automatic logging**, and **checkpointing**
- Cleaner code → faster experimentation

It replaces the hand-written 5-step loop from Section 2.3 with a `LightningModule` that defines the model, loss, and optimizer once.

### 16.3 Deployment and inference

- Switch to **`model.eval()`** before inference
- Optimize for **speed and memory**
- Export the model to run outside the training environment (production, mobile, edge), for example via **ONNX**

> **Note:** the course material lists TorchScript as an export route. PyTorch's documentation now marks TorchScript as deprecated in favor of `torch.export`, so check the current docs before choosing an export format.

---

## Stage 5 self-check

Try each question first, then open the answer.

**1. A 16×16 image goes through `Conv2d(kernel_size=5, padding=2)` and then `MaxPool2d(2)`. What is the output size?**

<details>
<summary>Show answer</summary>

The convolution keeps 16×16, because (16 + 4 − 5)/1 + 1 = 16. Pooling then halves it to 8×8.

</details>

**2. What is the weight shape of `nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)`?**

<details>
<summary>Show answer</summary>

`(32, 3, 3, 3)`: 32 kernels, each of size (3, 3, 3).

</details>

**3. What does a residual block compute, and why does it help?**

<details>
<summary>Show answer</summary>

`output = F(x) + x`. The skip connection gives gradients a direct path to early layers, and the block can easily learn a near-identity mapping (F(x) ≈ 0), which avoids degradation in very deep networks.

</details>

**4. List the steps to adapt a pretrained ResNet18 to 7 classes.**

<details>
<summary>Show answer</summary>

Load it with `weights=models.ResNet18_Weights.DEFAULT`, freeze all parameters (`requires_grad=False`), replace `model.fc` with `nn.Linear(model.fc.in_features, 7)`, and give the optimizer only `model.fc.parameters()`.

</details>

**5. When training on a GPU, which tensors must be moved to the device?**

<details>
<summary>Show answer</summary>

The model and both the features and the labels. For pure prediction, only the inputs (move predictions or labels to the same device if we compare them).

</details>


---

# Quick Reference

## Which setup for which task?

| Task | Output layer | Loss | Getting the prediction |
|---|---|---|---|
| Regression (one or several numbers) | `nn.Linear(in, num_outputs)` | `nn.MSELoss()` | The raw output |
| Binary classification | `nn.Linear(in, 1)` | `nn.BCEWithLogitsLoss()` | `sigmoid`, then threshold (usually 0.5) |
| Multi-class classification | `nn.Linear(in, K)` | `nn.CrossEntropyLoss()` | `argmax` of the logits (`softmax` if you need probabilities) |
| Multi-class with hidden layers (e.g., MNIST) | `nn.Linear(in, hidden)` + ReLU (several layers via `nn.ModuleList`), then `nn.Linear(hidden, K)` | `nn.CrossEntropyLoss()` | `argmax` of the logits |
| Image classification with a CNN | `[Conv2d → ReLU → MaxPool2d] × N`, flatten, then `nn.Linear(features, K)` | `nn.CrossEntropyLoss()` | `argmax` of the logits |
| Transfer learning (pretrained backbone) | Replace `model.fc` with `nn.Linear(model.fc.in_features, K)` | `nn.CrossEntropyLoss()` | `argmax` of the logits |

## Key rules

- **The training loop:** Forward → Loss → Reset (`zero_grad`) → Backward → Step.
- **Return logits from the model.** Let `BCEWithLogitsLoss` / `CrossEntropyLoss` apply sigmoid / softmax internally.
- **Never apply softmax before `CrossEntropyLoss`.**
- **Cross-entropy = negative log-likelihood.** Minimizing it is maximum likelihood estimation.
- **Why not MSE for classification?** With a sigmoid output, its gradients vanish when the model is confidently wrong.
- **A hidden layer needs a non-linear activation** (e.g., ReLU). Without it, stacked layers collapse into one linear map.
- **`loss.backward()` computes every gradient** via the chain rule; `optimizer.step()` applies them.
- **Vanishing gradients:** use ReLU, careful initialization (Part 11), and residual connections or batch normalization; momentum and adaptive optimizers (Part 12) help too.
- **`model.train()` vs. `model.eval()`:** dropout and batch normalization behave differently in each mode. Switch before training and before validation or inference.
- **Pick the initialization by activation:** Xavier for `tanh`/`sigmoid`, He for `ReLU`. PyTorch's default is already scaled by $1/\sqrt{L_{in}}$.
- **Dropout:** `nn.Dropout(p)` goes between hidden layers, never on the output layer. `p` is the probability of *dropping* a neuron.
- **Use `nn.ModuleList`** for layers built in a loop, so their parameters are registered with the model.
- **Transfer learning:** freeze the backbone (`requires_grad=False`), replace the output layer, and optimize only the new parameters.
- **GPU:** move the model and each batch with `.to(device)`. Labels must be on the GPU for training; for pure prediction, only the inputs.

## Neural network concept map (Parts 8 and 10)

| Concept | Key idea | PyTorch tool |
|---|---|---|
| **Multiple inputs** | Flatten images into feature vectors | `x.view(x.size(0), -1)` |
| **Dense hidden layers** | Every input connects to every neuron | `nn.Linear(in, hidden)` |
| **Forward propagation** | Linear → activation → linear → logits | `forward()` method |
| **Multiclass output** | One neuron per class, no softmax at the output | `nn.Linear(hidden, num_classes)` |
| **Loss function** | Softmax + NLL combined, numerically stable | `nn.CrossEntropyLoss` |
| **Backpropagation** | Chain rule, reuses gradients layer by layer | `loss.backward()` |
| **Vanishing gradients** | Small derivatives compound exponentially | ReLU, Adam, ResNet, BatchNorm |

## Deep networks, regularization, CNNs, and tooling (Parts 9 and 11 to 16)

| Topic | Core idea | PyTorch tool | Part |
|---|---|---|---:|
| **Deep NN** | More layers → hierarchical features | Stacked `nn.Linear` + activations | 9 |
| **Dynamic depth** | Build layers programmatically | `nn.ModuleList()` | 9 |
| **Symmetry breaking** | Random init → neurons learn different features | Default `nn.Linear` init | 11 |
| **Scaling rule** | Scale by $1/\sqrt{L_{in}}$ to prevent vanishing gradients | Built into the PyTorch default | 11 |
| **Xavier init** | For `tanh`; uses $L_{in} + L_{out}$ | `nn.init.xavier_uniform_` | 11 |
| **He init** | For `ReLU`; accounts for zeroed negatives | `nn.init.kaiming_uniform_` | 11 |
| **Batch norm** | Normalize per mini-batch before the activation | `nn.BatchNorm1d` | 11 |
| **Momentum** | Accumulate past gradients for smoother updates | `SGD(momentum=0.9)` | 12 |
| **Dropout** | Randomly disable neurons to prevent overfitting | `nn.Dropout(p)` | 13 |
| **Train vs. eval** | Dropout and batch norm differ by phase | `model.train()` / `model.eval()` | 11, 13 |
| **Convolution** | Kernels slide over images to detect local features | `nn.Conv2d` | 14 |
| **Activation + pooling** | Non-linearity, downsampling, translation invariance | `nn.ReLU`, `nn.MaxPool2d` | 14 |
| **Multiple channels** | Each output channel has its own kernel per input channel | `in_channels`, `out_channels` | 14 |
| **CNN** | Stack conv → ReLU → pool blocks, then flatten and classify | `nn.Sequential`, `nn.Linear` | 14 |
| **MNIST CNN** | Feature map 32×4×4 → 512 → 10 classes | `x.view(x.size(0), -1)` | 14 |
| **ResNet** | Skip connections address vanishing gradients and degradation | Residual blocks | 15 |
| **Transfer learning** | Freeze the backbone, replace the output layer | `torchvision.models`, `requires_grad=False` | 15 |
| **GPU** | Parallel acceleration for training and inference | `.to(device)`, `torch.cuda` | 16 |
| **Lightning** | Abstract the training boilerplate | `pytorch_lightning.LightningModule` | 16 |
| **Deployment** | Export for production | ONNX, `torch.export` (TorchScript is deprecated) | 16 |

**The pipeline:** inputs → dense hidden layers → forward pass → softmax-based loss → backprop → parameter updates. The vanishing-gradient problem explains *why* modern activations and optimizers matter.

**The throughline:** *depth* gives the model power (Part 9); *initialization, batch normalization, and momentum* keep gradients flowing through that depth (Parts 11 and 12); *dropout* keeps the power from turning into overfitting (Part 13). Convolution extracts spatial features, activation and pooling add non-linearity and invariance, and CNNs stack these blocks (Part 14). ResNets enable extreme depth, and pretrained models and GPUs make state-of-the-art results practical (Parts 15 and 16).
