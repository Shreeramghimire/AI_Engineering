**PyTorch** is a Python library for building and training machine learning models.



# Part 1

## Linear Regression in PyTorch

For simple linear regression, we could use Excel or plain math. But PyTorch gives:

- Fast math on lots of numbers, especially using GPUs

- Automatic gradients: it figures out how to adjust weights for you

- Efficient handling of big datasets

**Two Big Ideas**
 1.  Tensors, like NumPy arrays, but can run on GPUs.

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
```

2. Autograd, PyTorch automatically computes gradients.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # tensor(4.)  ← dy/dx = 2x = 4
```

**Training Linear Regression in PyTorch**

**The Core Idea**

Linear regression finds the best line:

y=wx+b

y = what we predict (e.g., ice cream sales)

x = input (e.g., temperature)

w = weight/slope

b = bias/intercept

**Training** = finding the best w and b so the line fits the data.

**Why Learn It in PyTorch?**

- It's the "Hello World" of ML

- The same workflow applies to deep neural networks — just swap in a bigger model

- You learn tensors, nn.Module, loss functions, optimizers, and the training loop, all of which transfer to CNNs, RNNs, Transformers

**The Training Loop (5 Steps)**

```text
Forward → Loss → Backward → Step → Reset → Repeat
```
1. Forward:	Make a prediction
   ```python
   y_pred = model(X)
   ```
 
2. Loss:	Measure how wrong
   ```python
   loss = criterion(y_pred, y)
   ```
3. Backward:	Compute gradients
   ```python
   loss.backward()
   ```
4. Step:	Update weights
   ```python
   optimizer.step()
   ```
5. Reset:	Clear old gradients
 ```python
 optimizer.zero_grad()
 ```

## Multi-Output Linear Regression

**What is Multi-Output Linear Regression?**

**Single-output linear regression** predicts **one number**:

$$y = wx + b$$

**Multi-output linear regression** predicts **several numbers at once**:

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

Example: Predict both **weight** and **sweetness** of a cake from ingredients → two outputs.

The model doesn't just give one answer; it gives a **vector of answers**, all at once.

---

| Concept | Analogy |
|---|---|
| Input features | Ingredients you put in |
| Output 1 (weight) | How much the cake weighs |
| Output 2 (sweetness) | How sweet the cake tastes |
| Weight matrix $W$ | Table showing how each ingredient affects each outcome |
| Bias vector $\mathbf{b}$ | Little extra nudge to make each prediction more accurate |
| Training | Tasting the cake, comparing to the ideal, adjusting the recipe |

The model learns to perfect **all predictions together**, just like perfecting a recipe to get the best cake every time.

---

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

Where:

- $\mathbf{x}$ = input vector (e.g., $[flour, sugar, eggs]$)
- $W$ = weight **matrix** (shape: `num_features × num_outputs`)
- $\mathbf{b}$ = bias **vector** (shape: `num_outputs`)
- $\mathbf{y}$ = output vector (e.g., $[weight, sweetness]$)

**Concrete example:**

Suppose:

- Inputs: 3 features (flour, sugar, eggs)
- Outputs: 2 targets (weight, sweetness)

Then:

- $W$ is a **3 × 2 matrix**
- $\mathbf{b}$ is a **2-element vector**
- $\mathbf{y}$ is a **2-element vector**

$$
W =
\begin{bmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22} \\
w_{31} & w_{32}
\end{bmatrix},
\qquad
\mathbf{b} =
\begin{bmatrix}
b_1 \\
b_2
\end{bmatrix}
$$

Each column of $W$ corresponds to one output. Each row corresponds to one input feature.

---

## The Cost Function

For single-output MSE:

$$\text{Loss} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

For multi-output, we **sum the squared errors across all outputs**:

$$\text{Loss} = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M}(y_{ij} - \hat{y}_{ij})^2$$

Where:

- $N$ = number of samples
- $M$ = number of outputs

**Key idea:** The model improves **all predictions together** — one aggregated loss drives learning for every output.

---

### In PyTorch

**The Model**

```python
import torch
import torch.nn as nn
```

**3 input features → 2 outputs**

```python
model = nn.Linear(in_features=3, out_features=2)
```

**The Loss**

```python
criterion = nn.MSELoss()
```

**Complete Training Example**

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

## Linear Classifiers and Decision Boundaries

Imagine we have fruits on a table: apples and oranges. We want to separate them automatically.

A **linear classifier** does exactly that: it draws a straight line that splits the fruits into two groups based on their features (like color or size).

- Fruit on one side → Apple
- Fruit on the other side → Orange

That line is called the **decision boundary**.

A decision boundary is the line that separates classes:

| Dimensions | Boundary |
|---|---|
| 2D | A line |
| 3D | A plane |
| Higher dimensions | A hyperplane |

The classifier uses a simple formula to decide which side a point belongs to.

---

## The Linear Function

Every linear classifier computes a **score**:

$$Z = W \cdot X + B$$

Where:

| Symbol | Meaning |
|---|---|
| $X$ | Feature vector (e.g., color, size) |
| $W$ | Weight vector (importance of each feature) |
| $B$ | Bias (shifts the boundary) |
| $Z$ | Score used to decide class |

**Example:** If $Z > 0$ → Class 1 (apple). If $Z \leq 0$ → Class 0 (orange).

---

### From Scores to Class Labels

The linear function gives a **continuous score** (any number). To get a class label, we apply a **threshold**:

If Z > 0 → Class 1
Else → Class 0

**Problem:** This gives a hard yes/no answer — no confidence.

- "This is an apple." 
- But how sure are we? 

---

### Logistic Regression and the Sigmoid Function

**Logistic regression** solves this by passing the score $Z$ through a **sigmoid function**:

$$\sigma(Z) = \frac{1}{1 + e^{-Z}}$$

This converts any score into a **probability between 0 and 1**.

| $Z$ | $\sigma(Z)$ | Meaning |
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

---

## Hard Decision vs. Soft Probability

| | Linear Classifier | Logistic Regression |
|---|---|---|
| Output | Class label (0 or 1) | Probability (0 to 1) |
| Confidence |  No |  Yes |
| Boundary | Hard line | Smooth curve |
| Example | "Apple" | "85% apple" |

---

## Applying a Threshold

To convert probability back to a class label, we pick a threshold (usually **0.5**):

If probability ≥ 0.5 → Class 1

Else → Class 0

The threshold can be adjusted:

- Lower threshold → more Class 1 predictions (higher recall)
- Higher threshold → fewer Class 1 predictions (higher precision)

---

## Bernoulli Distribution

A **Bernoulli distribution** models a single experiment with two outcomes:

- **Heads (success)**: probability = θ
- **Tails (failure)**: probability = 1 − θ

Mathematically:

$$
P(x) = \theta^x (1-\theta)^{1-x}
$$

where $x \in \{0, 1\}$ (1 = heads, 0 = tails).

**Example:** If θ = 0.2, then:

- $P(\text{heads}) = 0.2$
- $P(\text{tails}) = 0.8$

---

**Likelihood of a Sequence**

Suppose we flip the coin 3 times and get **Heads, Heads, Tails** (H, H, T).

Since flips are independent, multiply the probabilities:

$$
P(H,H,T) = \theta \cdot \theta \cdot (1-\theta) = \theta^2 (1-\theta)
$$

General formula for $n$ flips with $k$ heads:

$$
L(\theta) = \theta^{k} (1-\theta)^{n-k}
$$

This $L(\theta)$ is called the **likelihood**; it tells us how probable the observed data is for a given θ.

---

**Why Use Log-Likelihood?**

Multiplying many small probabilities can underflow (become ~0), and it's harder to differentiate. Take the **log**:

$$
\log L(\theta) = k \log(\theta) + (n-k) \log(1-\theta)
$$

This turns **multiplication into addition**, much easier to work with. Crucially, the θ that maximizes $\log L$ also maximizes $L$ (log is monotonic).

---

## Maximum Likelihood Estimation (MLE)

We want the θ that makes the observed data most likely. Take the derivative and set it to 0:

$$
\frac{d}{d\theta} \log L = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0
$$

Solving gives:

$$
\boxed{\hat{\theta}_{MLE} = \frac{k}{n}}
$$

**In simple word:** the MLE is just the fraction of heads we observed! If we flipped 10 times and got 7 heads, θ̂ = 0.7.

---
 **PyTorch Example:**

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
**Cross-Entropy Loss:** Cross-entropy loss looks at the predicted probability of the right answer and uses logarithms to measure how close the prediction is to the truth. The smoother and more informative this score is, the easier it is for the model to improve by adjusting its guesses step by step, like climbing down a hill to find the lowest point (best prediction). 

Imagine we're playing a guessing game where you have to predict if a picture shows a cat or not. Cross-entropy loss is like a scorekeeper that tells us how well our guesses match the truth. If we confidently say "cat" when it's really a cat, we get a good score (low loss). But if we confidently say "cat" when it's not, we get a bad score (high loss). This loss uses a special formula that punishes wrong and overconfident guesses more harshly, helping the model learn better.

## Logistic regression in PyTorch: 
Logistic regression is a way for a computer to make decisions about things that have two possible outcomes, like yes or no, true or false, or cat or dog. Imagine you want to teach a computer to decide if an email is spam or not. Logistic regression helps the computer look at the email and give a probability, a number between 0 and 1, that tells how likely the email is spam. If the number is close to 1, the computer thinks it's probably spam; if it's close to 0, it's probably not.

Think of logistic regression like a smart gatekeeper who uses a smooth curve (called the sigmoid function) to decide how confident it is about each choice. Instead of making a hard yes/no decision right away, it gives a gentle "maybe" score that helps the computer learn better over time. This smooth scoring is important because it allows the computer to improve its guesses step by step, using feedback from mistakes, until it gets really good at sorting emails—or any other yes/no problem!

**Logistic Regression Predicts Probabilities**

Logistic regression uses the sigmoid function to map any input to a value between 0 and 1, representing the probability of belonging to class 1.

The sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Where $z = \mathbf{w}^T\mathbf{x}+b$ is a linear combination of the inputs and weights.

**Real-world example: Email Spam Detection**

We want to classify emails as spam (1) or not spam (0). Features might be:

- Number of links in the email

- Frequency of words like "free," "win," "urgent"

- Whether the sender is in your contacts

The model computes z=2.5 for a particular email. Applying sigmoid:

$$
\sigma(2.5)=\frac{1}{1+e^{-2.5}}\approx 0.924
$$

Interpretation: There's a 92.4% probability this email is spam.

| **Email** | **\(z\) (raw score)** | **\(\sigma(z)\)** | **Interpretation** |
|---|---:|---:|---|
| A | 3.0 | 0.953 | 95.3% spam |
| B | 0.0 | 0.500 | 50% spam (uncertain) |
| C | −2.0 | 0.119 | 11.9% spam (likely not spam) |

More real-world examples:

- Medical diagnosis: Probability a patient has a disease given symptoms

- Credit card fraud: Probability a transaction is fraudulent

- Customer churn: Probability a customer will cancel their subscription

- Ad click prediction: Probability a user will click on an ad

### MSE (why is it problematic?)

Mean Squared Error (MSE) is a way to measure how accurate a predictive model is by calculating the average squared distance between its predictions and the actual results.

Because it measures physical distance, it is the gold standard for **regression tasks** (predicting continuous numbers like house prices, temperature, or stock values) rather than **classification tasks**. 

Suppose we define loss as:

$$
\text{Loss} =
\begin{cases}
0, & \text{if prediction is correct} \\
1, & \text{if prediction is wrong}
\end{cases}
$$

**Example: loan approval**

| **Applicant** | **True Label** | **Predicted Probability** | **Error Count Loss** |
|---|---|---:|---|
| A | Approved (1) | 0.51 | 0 (rounded to 1, correct) |
| B | Approved (1) | 0.99 | 0 (correct) |
| C | Rejected (0) | 0.49 | 0 (rounded to 0, correct) |
| D | Rejected (0) | 0.01 | 0 (correct) |


Whether the model predicts 0.51 or 0.99 for applicant A, the error count is the same (0). There's no gradient telling the model "0.99 is better than 0.51." The model can't improve.

Worse, if applicant A were predicted at 0.49 (wrong), the loss jumps to 1 — but the gradient is zero everywhere except at the decision boundary. Gradient descent has nothing to follow.

## Cross-Entropy Loss vs. MSE

Imagine we're building a spam filter. For each email, our model outputs a **probability**:

- "90% sure this is spam" → prediction = 0.9
- "60% sure this is not spam" → prediction = 0.4 (since not-spam = 1 − 0.6)

But the true label is binary: **spam (1)** or **not spam (0)**.

So the question becomes: **How do we score the model's probability predictions against a 0/1 label?**

That's exactly what a **loss function** does.

---
Cross-entropy loss measures how "surprised" the model is by the true label.

For binary classification:

$$
\text{BCE}(y, \hat{y}) = -\big[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \big]
$$

where:
- $y$ = true label (0 or 1)
- $\hat{y}$ = predicted probability (between 0 and 1)

**Let's see it in action:**

| True label $y$ | Predicted $\hat{y}$ | Loss $-\log(\hat{y})$ | Meaning |
|---|---|---|---|
| 1 (spam) | 0.9 | 0.105 | Small loss — good guess |
| 1 (spam) | 0.5 | 0.693 | Medium loss — uncertain |
| 1 (spam) | 0.1 | 2.303 | **Big loss** — confident but wrong! |

Notice the **confident-but-wrong penalty** — that's the key feature.

---

MSE (mean squared error) is:

$$
\text{MSE} = (y - \hat{y})^2
$$

It looks reasonable, but there's a hidden problem: when you combine MSE with a **sigmoid** output (which is what classifiers use), the gradients become tiny.

Let's visualize the issue:

- With **sigmoid + MSE**, the gradient of loss w.r.t. the weights includes a factor $\sigma'(z) = \hat{y}(1-\hat{y})$.
- When the model is very wrong (e.g., $\hat{y} \approx 0$ but $y = 1$), $\sigma'(z) \approx 0$ → **gradient vanishes** → learning stalls.
- This creates **flat regions** in the loss surface.

With **sigmoid + cross-entropy**, the $\sigma'(z)$ term cancels out mathematically, giving clean gradients that don't vanish.

---

**Maximum Likelihood Estimation (MLE)** says: find the parameters θ that make the observed data most probable.

For binary labels modeled as Bernoulli:

$$
P(y \mid \hat{y}) = \hat{y}^y (1-\hat{y})^{1-y}
$$

Take the **log** (easier to work with):

$$
\log P(y \mid \hat{y}) = y \log(\hat{y}) + (1-y)\log(1-\hat{y})
$$

Now **negate** it (because we *minimize* loss, but MLE *maximizes* likelihood):

$$
\text{Loss} = -\big[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \big]
$$

**That's exactly cross-entropy loss!** So minimizing BCE = maximizing likelihood. They're the same thing.

---
PyTorch Code: MSE vs. BCE in Action

Let's compare both on a simple binary classification problem:

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

BCE converges faster and to a lower loss.

MSE learns more slowly, especially when predictions start off very wrong.

---

In practice, PyTorch lets you skip the explicit sigmoid by using *BCEWithLogitsLoss*, it's numerically more stable:

```python
# Instead of this:
model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
loss_fn = nn.BCELoss()

# Do this (recommended):
model = nn.Linear(2, 1)                # no sigmoid
loss_fn = nn.BCEWithLogitsLoss()       # applies sigmoid + BCE internally
```

## Applying Cross-Entropy Loss in PyTorch: Logistic Regression for Spam Detection

Imagine we're trying to teach a computer to decide if an email is spam or not. The computer guesses a probability, like saying "I'm 70% sure this email is spam." Cross-entropy loss is like a score that tells the computer how good or bad its guess is compared to the true answer (spam or not spam). If the guess is close to the truth, the score is low (which is good), and if it's far off, the score is high (which means the computer needs to learn more).

This score is special because it changes smoothly as the computer adjusts its guesses, like a gentle hill that guides the computer downhill toward better answers. This smoothness helps the computer learn step-by-step by following the slope of the hill (using gradients) to improve its guesses. In PyTorch, this process is done by creating a simple model that predicts probabilities, using cross-entropy loss to measure errors, and an optimizer to update the model's settings until it gets better at classifying emails correctly.

- **Cross-entropy loss** measures how well predicted probabilities match true class labels, providing a **smooth and continuous loss surface**.

- This smoothness allows **gradient-based optimization** methods like gradient descent to update model parameters efficiently **without stalling**.

Cross-entropy gives us that smooth hill. MSE can create flat regions, especially when predictions are very wrong, so the model stops learning.

Implementing Logistic Regression in PyTorch

A logistic regression model is:

1. A **linear layer** (computes $z = wX + b$)
2. Followed by a **sigmoid activation** (squashes $z$ into $[0, 1]$ to give a probability)

Then:

- **Binary cross-entropy loss** (`nn.BCELoss`) compares predicted probabilities with actual labels.
- An **optimizer** such as stochastic gradient descent (SGD) updates model parameters based on gradients.

### The Model in Code

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

## The Training Loop 

Every iteration does 5 things:

- Forward pass: make predictions

- Compute loss: compare predictions to true labels

- Zero gradients: clear old gradients

- Backward pass: compute new gradients

- Update parameters: nudge weights in the downhill direction

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

**Prediction (after training)**

After training, predicted probabilities are converted into class predictions using a threshold (commonly 0.5):

```python
with torch.no_grad():
    probabilities = model(X)
    predictions = (probabilities > 0.5).float()   # 1 if spam, 0 if not
```

- Probability > 0.5 → predict spam (1)

- Probability ≤ 0.5 → predict not spam (0)


## Advanced Optimization, Regularization, and Generalization in PyTorch


### 1. The Mountain-in-the-Fog Analogy


Imagine you're trying to find the fastest way down a mountain in the fog. The mountain is your model's **error**, and you want to reach the bottom (lowest error) as quickly and safely as possible. The **optimizer** is your guide, deciding which path to take and how big your steps should be.

- **Adam** is a smart guide who watches how steep the path is and adjusts your step size **for each foot separately**. If one foot is on a slippery slope, it takes smaller steps; if the other is on a gentle slope, it takes bigger steps.
- **RMSProp** looks at how steep the path has been **recently** and adjusts your step size to avoid big jumps — great for uneven terrain.
- **AdamW** is like Adam but better at keeping your shoes clean (**regularization**). It separates the cleaning from the walking, so your shoes last longer.

Keep this analogy in mind — everything below builds on it.

---

### 2. Optimizers: Adam, RMSProp, and AdamW

#### Adam (Adaptive Moment Estimation)

Adam adapts the learning rate **individually for each parameter** using running averages of gradients.

- Keeps a **first moment** (mean of gradients) and a **second moment** (mean of squared gradients).
- Divides the update by the square root of the second moment — so noisy parameters get smaller steps.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

- Pros: Fast convergence, robust to learning rate choice.
  
- Cons: Weight decay is coupled with gradient updates, which can hurt generalization.

#### RMSProp
RMSProp normalizes updates by maintaining a moving average of squared gradients.

- Doesn't use momentum like Adam, just a scaled gradient.

- Excellent for sequential models (RNNs, LSTMs) where gradients vary wildly.

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
```

- Pros: Stable in non-stationary settings.
  
- Cons: No momentum, can be slower than Adam on some tasks.

#### AdamW (Adam with Decoupled Weight Decay)

AdamW is like Adam, but weight decay is applied separately from the gradient update.

- In classic Adam + L2, decay gets scaled by the adaptive learning rate — which weakens its effect.

- AdamW decouples them, giving better regularization and generalization, especially in large networks.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

- Pros: Better generalization, current default for transformers.
  
- Cons: Slightly more hyperparameters to tune.

## Learning Rate Strategies

**Step Decay**

Reduce the learning rate by a factor every N epochs.

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

**Plateau Reduction**

Reduce when a monitored metric (e.g., validation loss) stops improving.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)
# In training loop:
scheduler.step(val_loss)
```
**Cosine / Gradual Decay**

Smoothly decay LR following a cosine curve — very popular in modern training.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```
**Warm-Up**

Start with a very small LR, then gradually increase. This stabilizes early training, especially in large models.

```python
def warmup_lambda(epoch):
    if epoch < 5:
        return (epoch + 1) / 5
    return 1.0
```
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

# Part 2

## Softmax function

In General words:

Imagine we're at a carnival booth where you throw darts at 10 targets labeled 0 through 9. Each dart we throw produces a "score" for every target: some higher, some lower. But these scores are wild numbers: 2.3, -1.5, 8.7, etc. They're not easy to interpret.

Softmax is the booth operator who takes those raw scores and converts them into percentages that add up to 100%. If our scores were [2.3, -1.5, 8.7], softmax might say: "Target 0: 5%, Target 1: 0.5%, Target 2: 94.5%." Now we can say, "The model is 94.5% confident this is a 2."

The word "soft" means it doesn't just pick the winner (that would be hard max, or argmax). Instead, it gives every class a small slice of probability, weighted by how high its score was. The winner still gets the biggest slice.



Given raw scores (called **logits**) `z=[z1,z2,...,zK]`:

```math
\text{softmax}(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
```
Three things to notice:

- Exponentiate each score → makes everything positive and amplifies differences.

- Sum all the exponentials → this is the denominator.

- Divide each by the sum → every output is between 0 and 1, and all outputs sum to 1.

That last property is why softmax is used as a probability function.

**Why Exponentiate?**

- $e^x$ is always positive → no negative "probabilities."
- It exaggerates differences: if $z1=5$ and $z2=1$, then $e^5/e^1≈55$, so class 1 gets ~55× the probability of class 2.
- It's smooth and differentiable → perfect for gradient descent.

**Simple example:**
Imagine we train a neural network to look at an image and classify it into one of three classes (K = 3): Cat, Dog, or Bird.

1. The Inputs (K Arbitrary Logits)

The model looks at a picture and outputs the following raw scores (logits):
• Cat: 2.0
• Dog: 1.0
• Bird: -1.0

They don't add up to 1, and one value is negative.

2. How Softmax Transforms Them

The Softmax function does this in two steps:

1. Exponentiates the scores (\(e^{logit}\)): This turns all negative numbers into positive numbers and makes larger numbers stand out more.

	• $(\text{Cat} \rightarrow e^{2.0} \approx 7.39\)$

	• $(\text{Dog} \rightarrow e^{1.0} \approx 2.72\)$

	• $(\text{Bird} \rightarrow e^{-1.0} \approx 0.37\)$

	• Total Sum of Exponents = $(7.39 + 2.72 + 0.37 = \mathbf{10.48}\)$

3. Normalizes them: Divide each exponent by the total sum so they add up to 1.
   
	• Cat Probability: $7.39 / 10.48 = \mathbf{0.705}\$ (70.5%)

	• Dog Probability: $2.72 / 10.48 = \mathbf{0.260}\$ (26.0%)

	• Bird Probability: $0.37 / 10.48 = \mathbf{0.035}\$ (3.5%)

5. The Result

• Sum check: $(0.705 + 0.260 + 0.035 = \mathbf{1.0}\)$ (They sum to exactly 1).

• Order check: The original logit order was Cat (2.0) > Dog (1.0) > Bird (-1.0). The final probability order is Cat (70.5%) > Dog (26.0%) > Bird (3.5%). The order is perfectly preserved.

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

## Multi-Class Classification with Softmax in PyTorch

In binary classification, we have one output neuron that produces a single score, and we apply a sigmoid to squash it into a probability. In multi-class classification, we have one output neuron per class, each producing its own score. Softmax then converts these scores into a probability distribution across all classes.

### 1. Linear Equations Generate Logits for Each Class

For a network with K classes, each class k has its own weight vector wₖ and bias bₖ. Given an input vector x (features from the previous layer), the logit for class k is:

zₖ = wₖ · x + bₖ

Stacking all classes together:

z = Wx + b        where W is (K × D), z is (K,)

In PyTorch, this is simply a *nn.Linear* layer:

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

These raw outputs are called logits, unnormalized, unbounded real numbers. Each one measures "how strongly the model favors this class" relative to the others. They are not probabilities yet.

### 2. Softmax Converts Logits into Probabilities

Softmax transforms the logit vector z into a probability distribution:

softmax(z)ₖ = exp(zₖ) / Σⱼ exp(zⱼ)

**Key properties:**

- Every output is in (0, 1)

- All outputs sum to 1

- Larger logits → larger probabilities

- Differences between logits are preserved as ratios of probabilities

```python
probs = torch.softmax(logits, dim=1)
print(probs)
# tensor([[0.32, 0.07, 0.61]])
print(probs.sum(dim=1))   # tensor([1.])  ← sums to 1
```

**Why exp?**

- exp is always positive → no negative probabilities

- It's monotonic → ranking of logits is preserved

- It amplifies differences (logit gap of 2 → probability ratio of e² ≈ 7.4)

### 3. Argmax Selects the Predicted Class

The predicted class is the one with the highest logit; equivalently, the highest probability (softmax is monotonic):

```python
predicted_class = torch.argmax(logits, dim=1)
print(predicted_class)   # tensor([2])  ← class index 2 has the largest logit
```
argmax returns the index, not the value. In this example, logit 0.87 was largest, so class 2 wins.

**Putting It All Together: A Full PyTorch Example**

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

**Training: Use CrossEntropyLoss, Not Softmax + NLL**

PyTorch's nn.CrossEntropyLoss combines log_softmax + negative log-likelihood in one numerically stable operation. So during training we feed it raw logits:



