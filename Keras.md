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

- Input layer – One node per input feature. It doesn't perform computation; it just distributes the input to the next layer.

- Hidden layers – One or more layers of neurons that transform the input. "Deep" learning refers to having many hidden layers. Each layer learns increasingly abstract representations (e.g., edges → shapes → objects in image recognition).

- Output layer – Produces the final prediction. The number of nodes depends on the task: one for binary classification, one per class for multi-class, one per value for regression.

