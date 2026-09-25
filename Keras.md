## Biological Neuron Structure
The main body of a neuron is the soma, and the extensive network of arms that stick out of the body are called dendrites. The long arm that sticks out of the soma in the other direction is called the axon. Whiskers at the end of the axon are called the synapses.

A biological neuron has these parts:

**Soma (cell body):** Contains the nucleus and performs the cell's metabolic functions. It integrates incoming signals.

**Dendrites:** Branch-like extensions that receive signals from other neurons. Their branching structure allows a single neuron to receive input from thousands of others.

**Axon:** A long, thin projection that carries the neuron's output signal away from the soma toward other neurons.

**Synapses.** The junctions at the end of the axon (and on dendrites) where signals are transmitted from one neuron to the next.

**Signal flow:** Dendrites receive electrical impulses that carry information from synapses of other adjoining neurons. Dendrites carry the impulses to the soma. In the nucleus, electrical impulses are processed by combining them, and then they are passed on to the axon. The axon carries the processed information to the synapses, and the output of this neuron becomes the input to thousands of other neurons.

1. Reception: Neurotransmitters released from a presynaptic neuron bind to receptors on the dendrites, creating electrical changes (postsynaptic potentials).

2. Integration: These electrical signals travel to the soma, where they are summed. If the combined signal exceeds a threshold, the neuron "fires."

3. Transmission: An action potential (electrical spike) travels down the axon.

4. Output: At the synapses, the electrical signal triggers the release of neurotransmitters, passing the message to the next neurons.

This "all-or-nothing" firing and the summation of inputs is the biological basis for the activation function in artificial neurons (e.g., a step function or sigmoid that decides whether a node "fires").

**Learning in the brain** occurs by repeatedly activating certain neural connections over others, and this reinforces those connections.

This is the principle behind Hebbian learning, often summarized as "neurons that fire together, wire together." When two neurons are repeatedly active at the same time, the synapse between them strengthens — a process called long-term potentiation (LTP). Conversely, unused connections weaken (long-term depression).

In artificial neural networks, this idea is mirrored by weight adjustment: connections that contribute to correct outputs are strengthened (weights increased), and those that contribute to errors are weakened. The difference is that biological learning is local and unsupervised, while artificial networks typically use a global error signal (backpropagation) to adjust weights.

## Artificial Neuron

An artificial neuron behaves in the same way as a biological neuron.
An artificial neuron (also called a node or unit) performs a simplified version of the biological process:

1. **Inputs** (x₁, x₂, …, xₙ) analogous to signals from dendrites.

2. **Weights** (w₁, w₂, …, wₙ) analogous to synaptic strengths.

3. **Summation:** multiply each input by its weight and add them up: z = Σ(wᵢxᵢ) + b, where b is a bias term.

4. **Activation function:** applies a non-linear transformation (e.g., ReLU, sigmoid) to decide the output.

5. **Output:** passed to the next layer.

### Layers of a Neural Network

A feedforward neural network is organized into layers:

- **Input layer:** One node per input feature. It doesn't perform computation; it just distributes the input to the next layer.

- **Hidden layers:** One or more layers of neurons that transform the input. "Deep" learning refers to having many hidden layers. Each layer learns increasingly abstract representations (e.g., edges → shapes → objects in image recognition).

- **Output layer:** Produces the final prediction. The number of nodes depends on the task: one for binary classification, one per class for multi-class, one per value for regression.

### Forward Propagation
Forward propagation is the process through which data passes through layers of neurons in a neural network from the input layer to the output layer.

- Input values are multiplied by the weights of the first hidden layer, summed, and passed through an activation function.

- The outputs of that layer become the inputs to the next layer.

- This repeats until the output layer produces the final result.

In mathematical terms, for each layer l:
a⁽ˡ⁾ = f(W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)
where f is the activation function, W is the weight matrix, b is the bias vector, and a is the activation (output) of the layer.

Forward propagation is used both during **training** (to compute the prediction before calculating the loss) and during **inference** (to make predictions on new data).

### Computing Network Output

This is the core idea of a parametric model: once the **weights** and **biases** are learned (during training), the network becomes a deterministic function that maps any input to an output. No further learning is needed for inference.

Weights and biases are the core learnable parameters that allow a neural network to learn patterns, make decisions, and improve its accuracy from data. 

Weights determine the strength or importance of each input feature coming into a neuron. Mathematically: Each incoming input (x) is multiplied by its corresponding weight (w). 

Bias is an extra constant value added to the weighted inputs. It shifts the final result of a neuron up or down regardless of the input values. Like a y-intercept in a linear equation (y = mx + b), the bias (b) is added right after multiplying inputs by weights: z = w ⋅ x + b. Without bias, the network's functions would always be forced to pass through the origin (0,0), heavily limiting flexibility.

$$ \text{Output}=(\text{inputs}\times \text{weights})+\text{bias}\ $$

For example, a trained network for digit recognition (0–9) takes a 28×28 pixel image, performs forward propagation through its layers, and outputs a probability distribution over the 10 digits. The predicted digit is the one with the highest probability.

### Gradient Descent

Gradient descent is an iterative optimization algorithm for finding the minimum of a function. It is the workhorse of neural network training. The function being minimized is the loss function (also called the cost function), which measures how wrong the network's predictions are.

The algorithm works by:

1. Computing the gradient (slope) of the loss with respect to each weight and bias.

2. Moving the parameters in the opposite direction of the gradient (downhill) by a step proportional to the learning rate.

3. Repeating until the loss stops decreasing or a stopping criterion is met.

Mathematically: w ← w − α · ∂L/∂w, where α is the learning rate and ∂L/∂w is the gradient of the loss with respect to the weight.

### Learning Rate
A large learning rate (α) can lead to big steps and miss the minimum point. A small learning rate can result in extremely small steps and cause the algorithm to take a long time to find the minimum point.

**Too large:** The algorithm overshoots the minimum, possibly oscillating or diverging (loss increases instead of decreasing).

**Too small:** Convergence is extremely slow; training may take an impractical amount of time or get stuck in a plateau.

**Just right:** Steady, efficient convergence to a good minimum.

In reality, techniques like **learning rate schedules**, **Adam optimizer**, or **learning rate decay** are used to adapt the rate during training — starting larger for fast initial progress and shrinking for fine-tuning.

### The Training Loop
Neural networks train by initializing weights and biases randomly. Subsequently, we repeat the following process in a loop: forward propagation → calculate error → backpropagation → repeat until iterations/epochs are reached or the error is below the threshold.

The training loop at the heart of deep learning. Breaking it down:

1. Initialization: Weights are set to small random values (not zeros, because that would make all neurons in a layer identical). Common schemes: Xavier/Glorot, He initialization.

2. Forward propagation: Input data flows through the network to produce predictions.

3. Loss computation: The error between predictions and ground truth is measured (e.g., mean squared error, cross-entropy).

4. Backpropagation: The gradient of the loss with respect to each weight is computed using the chain rule, working backward from the output layer.

5. Weight update: Gradient descent adjusts weights and biases.

6. Repeat: One full pass through the training data is an epoch. Training stops when a maximum number of epochs is reached, or the loss falls below a threshold (or stops improving — early stopping).

### The Vanishing Gradient Problem

The vanishing gradient problem occurs when the error signal shrinks exponentially as it propagates backward through deep neural networks or **recurrent neural networks (RNNs)**, causing weights in earlier layers to stop updating.

**Cause:**

1. Saturating activation functions: Functions like sigmoid squash inputs into a small output range, producing small derivatives

2. Repeated multiplication in backpropagation: Backpropagation multiplies the derivatives of activation functions and weights across every layer; since these values are typically less than 1, the product shrinks as it moves backward

3. Poor weight initialization: Starting with initial weights that are too small accelerates the exponential decay of the gradient signal

**Consequences**

- Gradients (the values used to adjust network weights via backpropagation) become extremely close to zero

- Early layers freeze in place and fail to learn meaningful feature representations

- Sequential models (RNNs) lose the ability to connect information from early time steps to later predictions

Example: In a simple two-neuron network, the error gradient with respect to an early weight (e.g., w1) is very small, because backpropagation keeps multiplying factors less than one together; gradients shrink further with each layer moved backward.

**Common Solutions**

Non-saturating activations: Switch to ReLU, which maintains a derivative of 1 for positive inputs

Better initialization: Use Xavier or He (Kaiming) initialization to keep variance controlled across layers

Architectural shortcuts: Use ResNets (skip connections) or LSTMs/GRUs for sequential data, letting gradients flow unimpeded

Normalization: Apply batch normalization to stabilize the distribution of layer inputs

**What is initialization?**
Initialization includes setting the initial values of weights for the models, neural networks, or other deep learning architectures.

**Xavier initialization**, also known as Glorot initialization, is a technique designed to keep the variance of activations and gradients relatively constant across all layers of a deep neural network, preventing gradients from becoming too small (vanishing) or too large (exploding).

Core objective: Ensure that the variance of the outputs of a layer equals the variance of its inputs.

It achieves balance by drawing initial weights randomly from a distribution with a mean of 0 and a specific variance based on the layer's number of input and output units (fan-in and fan-out).

Depending on your framework, Xavier initialization samples weights from one of two distributions: 

- Uniform Distribution:

$$W\sim U\left(-\sqrt{\frac{6}{n_{in}+n_{out}}},\sqrt{\frac{6}{n_{in}+n_{out}}}\right)\$$

- Normal Distribution:

$$ W\sim N\left(0,\sigma ^{2}\right)$$  

$${where}\quad \sigma =\sqrt{\frac{2}{n_{in}+n_{out}}}\]$$

$n_{in}\(fan-in)$: The number of input units to the layer. $n_{out}\(fan-out)$ : The number of output units from the layer. 

It is mathematically optimized for activations that are linear or symmetric around zero, such as **tanh** and **sigmoid**.

**He initialization**, also known as *Kaiming initialization*, is a technique designed specifically for deep neural networks that use asymmetric, rectified activation functions like ReLU (Rectified Linear Unit) and its variants.

Introduced by **Kaiming He et al. in 2015**, it solves the problem where Xavier initialization underestimates the weight variance needed when half of the neurons are deactivated (outputting zero) by ReLU.

**Why Xavier Fails on ReLU**

The ReLU Drop: Because ReLU maps all negative inputs to zero, roughly 50% of the neurons in a layer turn off at any given time during training.

Variance Halving: This deactivation slashes the variance of the layer's output signal by half.

The Consequence: If you use Xavier initialization with ReLU, the signal variance will still decay exponentially across many deep layers, leading back to the vanishing gradient problem.

**He** initialization counteracts the 50% loss of signal by doubling the variance of the initial weights compared to Xavier. Instead of factoring in both inputs and outputs $\[n_{in}\]$ and $[n_{out}\]$, it focuses primarily on the number of incoming connections $\[n_{in}\]$ or (fan-in). 

Weights are randomly sampled from a distribution centered around 0 with a boosted variance:

Normal Distribution:

$$W\sim N\left(0,\sigma ^{2}\right)\quad$$ 

$${where}\quad \sigma =\sqrt{\frac{2}{n_{in}}}\$$


Uniform Distribution: 

$$W\sim U\left(-\sqrt{\frac{6}{n_{in}}},\sqrt{\frac{6}{n_{in}}}\right)\$$


**He vs. Xavier** 

| **Feature** | **Xavier (Glorot) Initialization** | **He (Kaiming) Initialization** |
|---|---|---|
| **Optimized For** | Linear or symmetric activations (**tanh**, **sigmoid**) | Rectified, asymmetric activations (**ReLU**, **Leaky ReLU**) |
| **Normal Variance $$\(\sigma^2\)$$** | $$(\frac{2}{n_{in}+n_{out}}\)$$ | $$(\frac{2}{n_{in}}\)$$ |
| **Core Assumption** | Activations are approximately linear around zero. | Approximately half of the activations are zeroed out by ReLU. |

**What is ResNets?**

The Concept of skip connections (or shortcut connections), which allow training of incredibly deep neural networks without suffering from the vanishing gradient problem. Introduced by Kaiming He et al. in 2015. 

**Fundamentals:**

- Skip Connections: ResNet uses shortcut paths (or skip connections) that allow the input of a layer to bypass one or more intermediate layers and connect directly to a later layer. It provides an uninterrupted path for gradients to flow backward during training, preventing signal loss.

- Residual Learning: Instead of learning the absolute mapping from input to output, the layers learn a residual function, the difference or change needed relative to the input. It made training networks with 50, 101, 152, or even over 1,000 layers practical and effective.

- Solves Degradation: Adding more layers to traditional neural networks often leads to higher training error, known as the degradation problem. ResNets ensure that deeper models perform at least as well as shallower ones.

- Common variants include ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152, where numbers indicate the layer depth.

-  Widely used as a backbone for image classification, object detection, instance segmentation, and medical image analysis.

**What are LSTMs and GRUs?**

LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units) are specialized types of Recurrent Neural Networks (RNNs) designed to process sequential data, such as text, speech, time-series data, and video.

While ResNets solved the vanishing gradient problem for deep spatial networks (images), LSTMs and GRUs were invented to solve the same problem for deep temporal networks (sequences). Standard RNNs struggle to retain information from many steps back because gradients shrink exponentially over time; LSTMs and GRUs use an internal mechanism called gates to regulate the flow of information and maintain a long-term memory.

| **Feature** | **LSTM (Long Short-Term Memory)** | **GRU (Gated Recurrent Unit)** |
|---|---|---|
| **Year Introduced** | 1997 (by Hochreiter & Schmidhuber) | 2014 (by Cho et al.) |
| **Internal States** | Uses **two states**: Cell State $\(c_t\)$ and Hidden State $\(h_t\)$ | Uses **one state**: Hidden State $\(h_t\)$ |
| **Number of Gates** | **Three gates**: Forget, Input, and Output gates | **Two gates**: Reset and Update gates |
| **Parameters** | More parameters (typically slower to train and requires more memory) | Fewer parameters (typically faster to train and more memory-efficient) |
| **Best Used For** | Problems where modeling complex long-term dependencies is important | Scenarios where computational efficiency and faster iteration are important |

**How LSTM Works:** The Three-Gate System

An LSTM maintains a Cell State, which acts like a conveyor belt carrying relevant information across long sequences. It modifies this conveyor belt using three distinct mathematical gates:

- Forget Gate: Decides what information from the past to throw away. It looks at the new input and the previous hidden state, outputting a number between 0 (completely discard) and 1 (completely keep).

- Input Gate: Decides what new information to store in the cell state. It determines which values to update and creates a vector of new candidate values.

- Output Gate: Decides what the next hidden state (and output) should be. It filters the updated cell state to only output the parts relevant to the current step.

**How GRU Works:** The Streamlined Alternative

The GRU is a newer, streamlined variation of the LSTM. It merges the cell state and hidden state, and reduces the architecture down to just two gates:

- Update Gate: Acts as a combined forget and input gate. It simultaneously determines how much past information to keep and how much new information to inject.

- Reset Gate: Decides how much of the past information to completely forget before processing the new input.

While LSTMs and GRUs revolutionized sequence modeling, they process data sequentially (step by step), which makes them difficult to parallelize on modern GPU hardware. Because of this limitation, they have largely been superseded by Transformers (like GPT and BERT) for massive Natural Language Processing tasks. However, LSTMs and GRUs remain incredibly useful for lightweight applications, real-time streaming data, edge devices, and specific industrial time-series forecasting where computational power is limited.


### Activation function:
An activation function is a mathematical formula applied to the output of every neuron in a neural network. Its primary purpose is to introduce non-linearity into the network, allowing it to learn complex, real-world patterns.

Without activation functions, no matter how many hundreds of layers you stack together, a neural network would just behave like a giant linear regression model, only capable of separating data with straight lines.

Inside a neural network, a single neuron performs a two-step calculation:

1. Linear Combination: It multiplies all inputs by their respective weights, adds them together, and adds a bias:

$$
z = (w_1 \times x_1) + (w_2 \times x_2) + \cdots + b
$$

2. Activation: It passes that final number $[z\]$ through the activation function $(f(z)\)$. This output is what gets sent forward to the next layer of neurons.

**Inputs $(x\)$** ───► **[ Weights & Bias ]** ───► **Linear Output $(z\)$** ───► **[ Activation Function ]** ───► **Final Output**

**Why Non-Linearity Matters**

Most real-world data is non-linear. Think of predicting housing prices, identifying faces, or driving autonomous cars—the relationships between variables are full of twists, curves, and sudden thresholds. 
A linear model cannot learn these curves. By applying a non-linear activation function (like ReLU or GELU), the network gains the ability to warp, bend, and shape its decision boundaries to fit virtually any complex dataset. This mathematical trait is known as the Universal **Approximation Theorem**.

**Modern Hidden Layer Standards**

These are used between the input and output layers to help the network learn internal features. 

- ReLU (Rectified Linear Unit): The reigning champion for computer vision. Formula: $\max(0, z)\$. It turns negative numbers into 0 and leaves positive numbers exactly as they are. It is incredibly fast and efficient.

- GELU (Gaussian Error Linear Unit): The standard choice for modern LLMs like GPT and BERT. It acts like a smooth version of ReLU, allowing a tiny bit of negative information to pass through, which helps with language understanding.


**Classic Functions** 

- Sigmoid / Tanh: Historically popular, they squash numbers into tight ranges (0 to 1 or -1 to 1). As discussed earlier, they are rarely used in hidden layers today because they cause the vanishing gradient problem.

  
**Output Layer Specialists**

- Softmax: Used for multi-class classification (e.g., deciding if an image is a cat, dog, or bird). It takes all output numbers and scales them into a probability distribution that adds up exactly to 1.0 (100%).

- Linear (No Activation): Used for regression tasks (e.g., predicting the exact numeric price of a house), where the output can be any arbitrary number.

**Sigmoid**

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Range: (0,1)

**Tanh**

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

Range: (-1,1)

**ReLU (Rectified Linear Unit)**

$$
f(x)=\max(0,x)
$$

Range: $(0,\infty)\$

**Leaky ReLU**

$$
f(x)=\max(\alpha x,x), \qquad \alpha \approx 0.01
$$

**Softmax**

$$
{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

- Converts: A vector into a probability distribution (the probabilities sum to 1)
- Used in: Multi-class classification output layers

**GELU / Swish**

- Smooth activation functions used in modern neural networks, including transformer architectures such as BERT and GPT.
- GELU:

$$
\{GELU}(x)=x.\Phi(x)
$$

where $\Phi(x)\$ is the **cumulative distribution function (CDF) of the standard Gaussian distribution**.

| **Function** | **Range** | **Best For** | **Downside** |
|---|---|---|---|
| **Sigmoid** | (0,1) | Binary output | Vanishing gradient |
| **Tanh** | (-1,1) | Hidden layers (older) | Vanishing gradient |
| **ReLU** | $\(0,\infty)\$ | Hidden layers (default) | Dying neurons |
| **Leaky ReLU** | $\(-\infty,\infty)\$ | Hidden layers | Extra hyperparameter |
| **Softmax** | (0,1), sums to 1 | Multi-class output | Only for output layer |
| **GELU** | $\approx(-0.17,\infty)\$ | Transformers | Computationally heavier |

## Deep Learning Libraries

| **Library** | **Level** | **Primary Use** | **Backed By** |
|---|---|---|---|
| **TensorFlow** | Low-level | Production deployment, large-scale systems | Google |
| **PyTorch** | Low-level | Research, experimentation, flexibility | Meta (Facebook) |
| **Keras** | High-level | Fast prototyping, ease of use | Now integrated into TensorFlow |

- **TensorFlow / PyTorch** = the **engine** (raw power, full control, more code)
- **Keras** = the **steering wheel and dashboard** (easy interface on top of the engine)

### TensorFlow

TensorFlow is used in the production of deep learning models and has a very large community of users.

TensorFlow was released by Google in 2015 and quickly became the industry standard for deploying models at scale. 

**Key characteristics**

- Production-ready: Tools like **TensorFlow Serving, TensorFlow Lite (mobile/edge), and TensorFlow.js** (browser) allow deployment across platforms.

- Static computation graph (in TF 1.x): You define the graph first, then run it. This made optimization easier but debugging harder.

- Eager execution (TF 2.x): Now executes operations immediately, making it feel more like PyTorch.

- Large community: Extensive documentation, tutorials, Stack Overflow answers, and pre-trained models.

- Ecosystem: TensorBoard (visualization), TFX (production pipelines), Keras (built-in high-level API).

**Strengths:** Deployment, scalability, tooling.

**Weaknesses:** Historically harder to learn; API changed significantly between 1.x and 2.x.

### PyTorch

PyTorch is based on the Torch framework in Lua and supports machine learning algorithms running on GPUs. It is widely used in academic research and machine learning experimentation.

PyTorch was released by **Meta (then Facebook)** in 2016 as a Python-based reimplementation of the **Torch** library, which was originally written in **Lua**.

**Key Characteristics**

- **Dynamic computation graph** – The graph is built on the fly as operations execute. This makes debugging intuitive because you can use standard Python debugging tools. It also works well with variable-length inputs, which are common in NLP.
  
- **Pythonic** – PyTorch feels similar to writing NumPy code, with automatic differentiation built in.
  
- **GPU support** – Provides CUDA integration for GPU acceleration. Data and models can be moved to a GPU using `.cuda()` or `.to(device)`.

- **Research-friendly** – Its flexibility makes it convenient for experimenting with novel architectures and custom models.

**Why Researchers Use PyTorch**

- Faster iteration cycles
  
- Easier implementation of custom layers and unconventional models
  
- Convenient debugging and experimentation
  
- Strong integration with the Python ecosystem

**Strengths and Weaknesses**

**Strengths:**
- Flexibility
  
- Research and experimentation

- Pythonic interface
  
- Easy debugging
  
- Strong GPU support

**Weaknesses:**
- Requires a good understanding of tensors, automatic differentiation, and training loops
  
- Can involve more code than high-level frameworks such as Keras
  
- Deployment tooling has historically been less straightforward, although the ecosystem has improved significantly

---

### Learning Curve

**PyTorch and TensorFlow can have a relatively steep learning curve compared with high-level frameworks such as Keras.**

Both frameworks require you to understand concepts such as:

- **Tensors** – Multi-dimensional arrays, similar to NumPy arrays, with support for GPU acceleration.
  
- **Computation graphs** – Structures that represent how operations are connected for automatic differentiation.
  
- **Device management** – Moving data and models between the CPU and GPU.
  
- **Training loops** – Controlling the forward pass, loss calculation, backpropagation, and optimizer updates.

**PyTorch vs. Keras Training**

A basic PyTorch training loop typically requires several explicit steps:

1. Perform the **forward pass**
2. Calculate the **loss**
3. Clear previous gradients
4. Perform **backpropagation**
5. Update model parameters

In Keras, much of this process can be handled automatically with:

```python
model.fit()
```

### Keras (The High-Level API)

*Keras is a high-level API for building deep learning models. It is popular because of its ease of use and simple syntax, which enables fast development and prototyping.*

Keras was created by **François Chollet** in 2015 with a philosophy centered around:

- **User-friendliness** – Consistent, simple APIs and clear error messages.
- **Modularity** – Models are built by combining independent, configurable layers.
- **Fast prototyping** – Go from an idea to a working model quickly.

Since **TensorFlow 2.0**, Keras has been the official high-level API of TensorFlow through `tf.keras`. Modern Keras can also work with multiple backends, including **TensorFlow, JAX, and PyTorch**.

---

#### Keras Abstraction

*Keras can build complex deep learning networks with only a few lines of code. It abstracts many low-level details that would otherwise need to be handled manually.*

| **Task** | **TensorFlow/PyTorch** | **Keras** |
|---|---|---|
| Define a layer | Manual weight initialization and forward-pass logic | `Dense(64, activation='relu')` |
| Build a model | Chain operations manually | `Sequential([...])` |
| Train | Write or configure a training loop | `model.fit(X, y, epochs=10)` |
| Evaluate | Manual metric computation | `model.evaluate(X, y)` |
| Predict | Execute the forward pass manually | `model.predict(X)` |

**Example: Same Model in Keras vs. PyTorch**

**Keras**

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=10)
```
**PyTorch**

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

model = Net()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
```

The Keras example requires significantly less code because Keras provides high-level abstractions for model construction, compilation, training, evaluation, and prediction.


## Data Preparation for Keras

### Preparing Data

*Before using Keras, you need to prepare your data and organize it in the appropriate format.*

Keras expects data in specific formats depending on the task:

- **Inputs $(X\)$** – NumPy arrays or tensors:
  - **Tabular data:** `(num_samples, num_features)`
  - **Images:** `(num_samples, height, width, channels)`
- **Targets $(y\)$** – NumPy arrays. The format depends on the task:
  - **Regression** – Continuous values, shape `(num_samples,)` or `(num_samples, 1)`.
  - **Binary classification** – 0 or 1, shape `(num_samples,)` or `(num_samples, 1)`.
  - **Multi-class classification** – One-hot encoded, shape `(num_samples, num_classes)`.

### Common Preprocessing Steps

- Handle missing values
- Normalize or standardize numerical features
- Encode categorical variables
- Split the data into training, validation, and test sets


### Predictors and Target

*A dataset can be divided into predictors and a target.*

This is the fundamental setup for **supervised learning**:

- **Predictors (features, (X))** – The input variables the model uses to make predictions.  
  Examples: age, income, education level.
- **Target (label, (y))** – The output variable the model tries to predict.  
  Example: whether a customer will churn (`yes`/`no`).

### In Python

```python
X = df.drop('target_column', axis=1)  # Predictors
y = df['target_column']               # Target
```

### Binary Encoding of Target

When using Keras for classification problems, the target variable needs to be represented in a numerical format.

Keras and most machine learning frameworks require numerical targets.

#### Binary Classification

For binary classification, the target is typically represented as 0 or 1.

For example:

Yes → 1

No  → 0

If the original labels are strings such as "yes"/"no" or "cat"/"dog", they must be encoded into numerical values.

#### Multi-Class Classification

For multi-class classification, the target can be one-hot encoded: a vector of 0s and 1s where the correct class is represented by 1.

For example, with three classes:

| **Original Label** | **One-Hot Encoded** |
| ------------------ | ------------------- |
| Cat                | `[1, 0, 0]`         |
| Dog                | `[0, 1, 0]`         |
| Bird               | `[0, 0, 1]`         |


This representation works naturally with a **softmax** output layer, which produces a probability for each class.

For binary classification, a **sigmoid** output layer is commonly used to produce the probability of the positive class.

### to_categorical() Function

The to_categorical() function from the Keras utilities package converts integer class labels into one-hot encoded vectors.

**Syntax**

```python
from tensorflow.keras.utils import to_categorical

y_encoded = to_categorical(y, num_classes=None)

```

**Example**
```python
from tensorflow.keras.utils import to_categorical
import numpy as np

y = np.array([0, 1, 2, 1, 0])

y_encoded = to_categorical(y)

print(y_encoded)

# [1. 0. 0.]
# [0. 1. 0.]
# [0. 0. 1.]
# [0. 1. 0.]
# [1. 0. 0.]
```

## Building Classification Model in Keras

**Step 1: Import libraries**
```python


 
