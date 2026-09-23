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


