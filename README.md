# Recurrent Neural Network with Attention mechanism
# Overview
This code implements a Long Short-Term Memory RNN encoder and decoder with attention. It does not make use of the pytorch nn.LSTM function and instead implements the formulas. In addition to the implementation, it contains a write up of the visualization for the attention mechanism and discusses selected plots.

# Table of Contents
1. Installation
2. Usage
3. Algorithms
4. Results
5. Contributors


# Installation
1. Make sure you have these running on your system:  
a. Python 3  
b. PyTorch  
c. NumPy  
d. Matplotlib  
e. NLTK (Natural Language Toolkit)  
f. GitHub  
Install the dependencies using: pip install torch numpy matplotlib nltk  

2. Follow the proper steps in the INSTALL_NOTES.md file

2. Clone the repository to your local machine:  
   **git clone https://github.com/nicoledeprey/MachineTranslation_hw5.git**

2. Navigate to the project directory:  
**cd hw5**


# Usage
Running seq2seq.py


1. Run the file with the following command:
**python seq2seq.py**

2. The arguments can be viewed by running:
**python seq2seq.py -h**


# Algorithms
## Long Short-Term Memory (LSTM)  
LSTM can be used to solve problems faced by the RNN model, such as, long term dependency problems and the vanishing and exploding gradient. LSTM makes use of three gates: forget gate, f, input gate, i, and output gate, o. LSTM also makes use of a cell state and candidate cell state to find the final output. A description of the LSTM Algorithm can be found in the MathDescription.pdf.


## Attention Visualization  
Attention is used to focus on different parts of the input at different steps. The attention mechanism computes attention scores for each element of the input sequence, indicating its relevance to the current decoding step. The attention scores are then normalized to create a probability distribution. A description of the attention decoder can be found in the MathDescription.pdf.

## Batch Training
Batch training in the context of neural networks involves updating model parameters based on the average gradient computed over a batch of training samples. 
Objective Function:
The objective is to minimize the average loss over a batch of training examples

Batch Gradient Descent:
The gradient of the objective function with respect to the parameters  is computed for the entire batch

In our code, we implement Mini-Batch Gradient Descent:
Mini-batch gradient descent strikes a balance between batch gradient descent and stochastic gradient descent. It computes the gradient and updates parameters based on a small random subset (mini-batch) of the training data



## Beam Search
Beam search is a heuristic search algorithm used for finding the most likely sequence of outputs in sequence generation tasks. It extends the concept of searching through a sequence of tokens while considering multiple candidate sequences simultaneously. Here's a mathematical description of how beam search works:

Initialization:

Initialize a set of candidate sequences C with a single start token <SOS>, each with an associated score (usually initialized to 0).

Iteration:
2. Repeat the following steps for a fixed number of decoding steps or until all candidate sequences have ended with an <EOS> token:
a. For each candidate sequence c in C, generate the set of possible next tokens V(c) using the model. Each token in V(c) has an associated conditional probability.

b. For each candidate sequence c, calculate the score of each possible extension v in V(c) by adding the log probability of v to the score of c. The score represents the cumulative log likelihood of the sequence.

c. Merge the candidate sequences C with the extended sequences from step (b). Sort the merged list by score and select the top-k sequences with the highest scores to form the new set of candidate sequences C.


Termination:
3. After decoding is complete or when a candidate sequence ends with an <EOS> token, select the sequence with the highest score as the final output sequence.


# Results
The code results should produce a BLEU Score of 0.4617316003241794.
This sequence-to-sequence with attention model offers a starting point for machine translation tasks and other applications where both the input and output are sequences. Users are encouraged to adapt and expand upon the base code to fit specific needs and challenges.

Impact on Speed:
Following the implementation of batch training, the training process was sped up compared to without batching. The increase in speed can be attributed to parallelism, vectorized operations, memory efficiency, and more stable gradients that batch training provides.

Following the implementation of the built-in PyTorch functions, the change in framework optimizations allowed the system to efficiently handle the batches of data, further contributing to the overall speedup compared to our original RNN.

With the addition of the Beam Search extension, the speed further increased. Beam search involves exploring multiple possible sequences during the decoding phase, which increases the time required for generating translations. The larger the beam size, the more alternative sequences are explored, leading to a longer inference time. While beam search introduces additional computations and longer inference time, it is possible to parallelize some of the operations during decoding. This can be done by evaluating multiple candidate sequences in parallel. By combining batching and beam search in parallel, this involves decoding multiple sequences in parallel across different batches.



# Contributors
This code was developed by Janvi Prasad, Nicole Deprey, and Hirtika Mirghani.
