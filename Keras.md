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

### Artificial Neuron

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

## Gradient Descent

Gradient descent is an iterative optimization algorithm for finding the minimum of a function. It is the workhorse of neural network training. The function being minimized is the loss function (also called the cost function), which measures how wrong the network's predictions are.

The algorithm works by:

1. Computing the gradient (slope) of the loss with respect to each weight and bias.

2. Moving the parameters in the opposite direction of the gradient (downhill) by a step proportional to the learning rate.

3. Repeating until the loss stops decreasing or a stopping criterion is met.

Mathematically: w ← w − α · ∂L/∂w, where α is the learning rate and ∂L/∂w is the gradient of the loss with respect to the weight.



