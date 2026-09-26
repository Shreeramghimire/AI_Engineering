# Part 1

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



