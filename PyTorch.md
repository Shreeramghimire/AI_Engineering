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

## In PyTorch

### The Model

```python
import torch
import torch.nn as nn
```

# 3 input features → 2 outputs

```python
model = nn.Linear(in_features=3, out_features=2)
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

