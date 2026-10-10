# PyTorch (From Linear Regression to Multi-Class Classification)

## Part 1 — PyTorch Foundations

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

## Part 2 — Regression: Predicting a Number

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

# ── 1. Data ──────────────────────────────────────────
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

# ── 2. Model ─────────────────────────────────────────
model = nn.Linear(in_features=3, out_features=2)

# ── 3. Loss ──────────────────────────────────────────
criterion = nn.MSELoss()

# ── 4. Optimizer ─────────────────────────────────────
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# ── 5. Training Loop ─────────────────────────────────
for epoch in range(5000):
    y_pred = model(X)                 # Forward → shape (5, 2)
    loss = criterion(y_pred, y)       # Aggregated loss over both outputs
    optimizer.zero_grad()             # Reset gradients
    loss.backward()                   # Backward
    optimizer.step()                  # Update W and b

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# ── 6. Predict ───────────────────────────────────────
model.eval()
with torch.no_grad():
    new_recipe = torch.tensor([[2.5, 1.5, 2.5]])
    prediction = model(new_recipe)
    print(f'Predicted [weight, sweetness]: {prediction.numpy()}')
```

> **What to expect:** the loss falls quickly at first and then keeps shrinking slowly. The targets are large (hundreds) and the learning rate is small, so the loss will still be well above zero after 5000 epochs. This is fine for a demo; scaling the data or raising the learning rate would speed things up.

---

## Part 3 — Binary Classification: Predicting a Category

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

## Part 4 — Probability Foundations: Where Classification Losses Come From

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

# --- Closed-form MLE ---
theta_mle = k / n
print(f"Closed-form MLE: {theta_mle.item():.4f}")  # 0.7

# --- Via gradient descent (how PyTorch really works) ---
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

## Part 5 — Loss Functions for Classification

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

# ---- Data: 4 samples, 2 features ----
X = torch.tensor([[1.0, 2.0],
                  [2.0, 1.0],
                  [-1.0, -2.0],
                  [-2.0, -1.0]])
y = torch.tensor([[1.0], [1.0], [0.0], [0.0]])  # binary labels

# ---- A tiny model: linear layer + sigmoid ----
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# ---- Try both losses ----
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

## Part 6 — Logistic Regression in PyTorch (Spam Detection)

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

## Part 7 — Multi-Class Classification with Softmax

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

## Part 8 — Better Training: Optimizers and Learning-Rate Strategies

So far we have used plain SGD. This part covers smarter optimizers and ways to adjust the learning rate during training.

### 8.1 The mountain-in-the-fog analogy

Imagine you're trying to find the fastest way down a mountain in the fog. The mountain is your model's **error**, and you want to reach the bottom (lowest error) as quickly and safely as possible. The **optimizer** is your guide, deciding which path to take and how big your steps should be.

- **Adam** is a smart guide who watches how steep the path is and adjusts your step size **for each foot separately**. If one foot is on a slippery slope, it takes smaller steps; if the other is on a gentle slope, it takes bigger steps.
- **RMSProp** looks at how steep the path has been **recently** and adjusts your step size to avoid big jumps — great for uneven terrain.
- **AdamW** is like Adam but better at keeping your shoes clean (**regularization**). It separates the cleaning from the walking, so your shoes last longer.

Keep this analogy in mind — everything below builds on it.

### 8.2 Optimizers: Adam, RMSProp, and AdamW

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

## 8.3 Learning-rate strategies

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

## Quick Reference

### Which setup for which task?

| Task | Output layer | Loss | Getting the prediction |
|---|---|---|---|
| Regression (one or several numbers) | `nn.Linear(in, num_outputs)` | `nn.MSELoss()` | The raw output |
| Binary classification | `nn.Linear(in, 1)` | `nn.BCEWithLogitsLoss()` | `sigmoid`, then threshold (usually 0.5) |
| Multi-class classification | `nn.Linear(in, K)` | `nn.CrossEntropyLoss()` | `argmax` of the logits (`softmax` if you need probabilities) |

### Key rules

- **The training loop:** Forward → Loss → Reset (`zero_grad`) → Backward → Step.
- **Return logits from the model.** Let `BCEWithLogitsLoss` / `CrossEntropyLoss` apply sigmoid / softmax internally.
- **Never apply softmax before `CrossEntropyLoss`.**
- **Cross-entropy = negative log-likelihood.** Minimizing it is maximum likelihood estimation.
- **Why not MSE for classification?** With a sigmoid output, its gradients vanish when the model is confidently wrong.
