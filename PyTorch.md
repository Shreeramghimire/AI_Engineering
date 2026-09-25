# Part 1

**Cross-Entropy Loss:** Cross-entropy loss looks at the predicted probability of the right answer and uses logarithms to measure how close the prediction is to the truth. The smoother and more informative this score is, the easier it is for the model to improve by adjusting its guesses step by step, like climbing down a hill to find the lowest point (best prediction). 

Imagine we're playing a guessing game where you have to predict if a picture shows a cat or not. Cross-entropy loss is like a scorekeeper that tells us how well our guesses match the truth. If we confidently say "cat" when it's really a cat, we get a good score (low loss). But if we confidently say "cat" when it's not, we get a bad score (high loss). This loss uses a special formula that punishes wrong and overconfident guesses more harshly, helping the model learn better.

## Logistic regression in PyTorch: 
Logistic regression is a way for a computer to make decisions about things that have two possible outcomes, like yes or no, true or false, or cat or dog. Imagine you want to teach a computer to decide if an email is spam or not. Logistic regression helps the computer look at the email and give a probability, a number between 0 and 1, that tells how likely the email is spam. If the number is close to 1, the computer thinks it's probably spam; if it's close to 0, it's probably not.

Think of logistic regression like a smart gatekeeper who uses a smooth curve (called the sigmoid function) to decide how confident it is about each choice. Instead of making a hard yes/no decision right away, it gives a gentle "maybe" score that helps the computer learn better over time. This smooth scoring is important because it allows the computer to improve its guesses step by step, using feedback from mistakes, until it gets really good at sorting emails—or any other yes/no problem!

**Logistic Regression Predicts Probabilities**

Logistic regression uses the sigmoid function to map any input to a value between 0 and 1, representing the probability of belonging to class 1.

