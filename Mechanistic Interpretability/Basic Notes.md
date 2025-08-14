> What if you simply go through the model, neuron by neuron, trying to understand each one and the connections between them? 

Basic definition of MI. 

Basically we want to be able to identify unknown safety problems (those "edge cases" that we can't anticipate in our training code), that come from either hindsight or "treacherous" features in the code. We can't study/anticipate these edge cases like we can for normal SWE, because code is not transparent for ML.
> Parameters for neural networks is equivalent to assembly instructions for normal coding.

Key features considered:
- Need methods to map neural network params to human understandable algos.
- Methods should be able to scale with enough human effort
- Should be applicable to standard neural networks.
- Should be able to discover unknown algos.
- Deep understanding of narrow region > shallow understanding of broader region.
# Resources
- [Very interesting tutorial on transformer specific interpretability, many resources](https://projects.illc.uva.nl/indeep/tutorial/)
# Potential Topics
- ROME and memory editing - causal intervention to follow the individual difference in layers after task-wise memory editing to find out how information travels in neural networks
- Superposition: 1 neuron stands for more than 1 representation/feature, ie a model can represent more features than it has dimensions.
	- Closer to the old Word2Vec idea of presenting embeddings as vector directions in an activation space that are additive
	- Neel thinks its an approximation, not a direct truth to think that model activations are sparse vectors
- SAEs - Neel thinks they are important in unsupervised settings, to understand whats happening initially, but eventually other models become more useful to solve problems
- Model Biology - high level understanding of what's happening inside a model for verifiable tasks
- Useful application - probes to detect harmful behavior in models

# EasyTransformer
- Can models figure out positional embeddings on their own? What happens if you don't give them positional information
- Usually a transformer has symmetry
- Causal attention pattern - things can only tend backwards, symmetry is broken because end has more positions to attend to than start neurons
	- So model might have some information, and it might be able to retrieve positional embeddings.
Idea - train model on sequence of random tokens, and have each token predict the token before. (instead of next token)
Use an MLP that has a start token, needs to learn uniform attention from start to Nth token, and predict 1/Nth of attention. 
![[no-position-hypothesis.png|800x400]]
His research Hypothesis
### Code changes
- turned positional embeddings off - manually set the `model.pos_embed.W_pos.data[:] = 0` (all layers in the pos_embed to 0)
- turned `model.pos_embed.W_pos.requires_grad = False` - `requires_grad` does automatic computation of gradients during backpropagation, so it shut them off for pos_embed layers so it the layers are not updated through training and remain set to 0.