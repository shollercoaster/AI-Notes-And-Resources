- Models need representations in the form of vectors of features, to be able to read text.
- How should the LM think? Words which frequently appear in **similar contexts** have **similar meaning**.

## How are these word vectors created?
- Vocabulary - acts as a lookup table. Special embedding attached to each word, check the lookup table. Ignore unknown words or give them an UNK token.

### One hot vectors
- 1 in the ith position for ith word, rest are 0.
- Not most efficient for large vocabularies
- Also don't capture meaning (embeddings don't represent semantic similarity)

### Count based methods
- Main idea: put info about context into word vectors
- These methods do it literally by manually putting info based on global corpus statistics.
- create a word-context matrix (rows = words, cols = context, so cells show whenever a word fit in a context)
- then reduce dimensionality using Singular Value Decomposition (to remove 0s)
- dot product / cosine similarity to find similarity between word/contexts
- Different methods arise from defining contexts and association / matrix elements in different ways:

#### Co-occurence counts
- Context = surrounding words in an L-sized window 
- Matrix elements (w, c) = number of times w appears in context c

#### Positive Pointwise Mutual Information
- same context
- ![alt text](https://github.com/shollercoaster/AI-Notes-And-Resources/blob/main/images/ppmi-formula.png "PPMI Formula")


#### Latent Semantic Analysis
- context is document d from a collection D
- uses TF IDF to compute similarity between document vectors 
- groups:
	- documents having similar sets of words
	- all words occurring in a document

### Word2Vec
- different from count-based: TEACHES word vectors to predict context of their surrounding words
- Word vectors are the learned params of the model, and goal is for each word vector to know the contexts it appears in
- Procedure:
	- take a sliding window over sentence, the surrounding words need to be predicted given central word
	- adjust vectors to increase probabilities

#### Objective Function: Negative Log-Likelihood
- All probabilities need to be multiplied (since they are independent)
- Word2Vec predicts context words, given the central word w<sub>t</sub>:

$\color{#88bd33}{\mbox{Likelihood}} \color{white}= L(\theta)= \prod\limits_{t=1}^T\prod\limits_{-m\le j \le m, j\neq 0}P(\color{#888}{w_{t+j}}|\color{#88bd33}{w_t}\color{white}, \theta), $

- ![alt text](https://github.com/shollercoaster/AI-Notes-And-Resources/blob/main/images/word2vec-formula.png "Word2Vec Formula")


$$
\color{#88bd33}{\mbox{Likelihood}} \color{white}= L(\theta)=
    \prod\limits_{t=1}^T\prod\limits_{-m\le j \le m, j\neq 0}P(\color{#888}{w_{t+j}}|\color{#88bd33}{w_t}\color{white}, \theta),
$$

- ![[word2vec-formula.png]]
Here $\theta$ refers to all variables being optimized. The loss function $J(\theta)$ is the average negative log likelihood:

$\color{#88bd33}{\mbox{Loss}} \color{white}= J(\theta)= \frac{-1}{T} logL(\theta) = \frac{-1}{T}\sum\limits_{t=1}^T\sum\limits_{-m \le j \le m, j\neq 0} log P(\color{#888}{w_{t+j}}|\color{#88bd33}{w_t}\color{white}, \theta),$

The first $\sum$ denotes going over all text, second $\sum$ donates sliding window, log part meaning computing probability of allcontext words given central.

##### Calculating $P(\color{#888}{w_{t+j}}\color{white}|\color{#88bd33}{w_t}\color{white}, \theta)$:
Each word w will have 2 vectors:
- $u_w$ when its a central word
- $x_w$ when its a context word

Now central word c and context word o, the probability of context word o is given by the ==SOFTMAX FUNCTION==! 

###### Softmax Function:
Found in a lot of places in NLP
- soft because all probabilities will be non-zero.
- Max because the higher $x_i$ will have a higher probability $p_i$.

$softmax(x_i) = \frac{exp(x_i)}{\sum\limits_{j=i}^N exp(x_j)}$

So applying this, the $p(o|c)$ becomes:

$p(o|c) = \frac{exp(x_o^Tu_c)}{\sum\limits_{w\in{V}}exp(x_w^Tu_c)}$

- Uses dot product to get cosine similarity
- Denominator normalizes it over all words to get the prob distribution

#### Training: using Gradient Descent
- Parameters $\theta$ to be learnt are the vectors $u_w$ and $x_w$, learnt using Gradient Descent:

$\theta^{new} = \theta^{old} - \alpha \nabla_{\theta} J(\theta).$

$\alpha$ is the learning rate.

- Updates are made 1 at a time, considering only 1 (centre word, context word) pair at once.

$\color{#88bd33}{\mbox{Loss}}\color{white} =J(\theta)= -\frac{1}{T}\log L(\theta)=-\frac{1}{T}\sum\limits_{t=1}^T\sum\limits_{-m\le j \le m, j\neq 0}\log P(\color{#888}{w_{t+j}}\color{black}|\color{#88bd33}{w_t}\color{white}, \theta)=\frac{1}{T} \sum\limits_{t=1}^T\sum\limits_{-m\le j \le m, j\neq 0} J_{t,j}(\theta).$

So let $J_{t,j}(\theta)=-\log P(\color{#888}{w_{t+j}}\color{white}|\color{#88bd33}{w_t}\color{white}, \theta)$.

Consider 1 pair in the sentence, 
"I saw a cute grey ==cat== playing in the garden."

Consider the pair (cute, cat) with cat being the centre word.

So the loss term $J_{t,j}(\theta)$ becomes:

$J_{t,j}(\theta)= -\log P(\color{#888}{cute}\color{black}|\color{#88bd33}{cat}\color{white}) = -\log \frac{\exp\color{#888}{x_{cute}^T}\color{#88bd33}{u_{cat}}}{\sum\limits_{w\in Voc}\exp{\color{#888}{x_w^T}\color{#88bd33}{u_{cat}} }} = -\color{#888}{x_{cute}^T}\color{#88bd33}{u_{cat}}\color{white}+ \log \sum\limits_{w\in Voc}\exp{\color{#888}{x_w^T}\color{#88bd33}{u_{cat}}}\color{white}{.}$

Only the parameters $x_{cute}$, $x_{grey}$, $x_{playing}$, $x_{in}$ (the vocabulary context words) and $u_{cat}$ will be updated at this step. 

##### Steps:
- Take dot products, get exponent and sum them all. You get: 

$\sum\limits_{w \in{V_o}}exp(x_w^Tu_{cat})$

- Get loss function for this one step

$J_{t,j}(\theta)=-x_{cute}^Tu_{cat} + log\sum\limits_{w \in_V}exp(x_w^Tu_{cat})$

- Find the gradient, make the update

$u_{cat} = u_{cat} - \alpha\frac{\nabla J_{t,j}(\theta)}{\nabla u_{cat}}$

$x_{w} = x_{w} - \alpha\frac{\nabla J_{t,j}(\theta)}{\nabla x_{w}}, \forall w \in V$

We minimize the cost function and so increase the cosine similarity between current context word $x_{cute}$ and centre word $u_{cat}$. And then at the same time, ==decrease== similarity between $u_{cat}$ and all other context words $x_w$.

#### Faster training: Negative Sampling
- Highly inefficient to make 1 update for a pair at once, as it takes time proportional to vocabulary size V.
- Update: instead of decreasing similarity for all other context words $x$, decrease it for a subset of other $x$, some selective negative samples.
- So instead of updating $u_{centre}$ and all $x_w$, we only update $u_{centre}$ and some $x_k$ for k in K negative samples.
- Instead of V + 1, we are updating K + 1 vectors.
- The new loss function becomes:

$J_{t,j}(\theta)=-\log\sigma(\color{#888}{x_{cute}^T}\color{#88bd33}{u_{cat}}\color{white}) -\sum\limits_{w\in \{w_{i_1},\dots, w_{i_K}\}}\log\sigma({-\color{#888}{x_w^T}\color{#88bd33}{u_{cat}}}\color{white}),$

where $\sigma$ is the sigmoid function and updation is only for K negative samples. Sigmoid function:

$\sigma(x)=\frac{1}{1+e^{-x}}$

Sigmoid of a negative function:

$\sigma(-x)=\frac{1}{1+e^{x}} = \frac{(1 + e^{x}) - 1}{1 + e^{x}} = 1 - \frac{1}{1+e^{-x}} = 1 - \sigma{(x)}$

So the loss function can be written as 
$J_{t,j}(\theta)=-\log\sigma(\color{#888}{x_{cute}^T}\color{#88bd33}{u_{cat}}\color{white}) -\sum\limits_{w\in \{w_{i_1},\dots, w_{i_K}\}}\log(1-\sigma({\color{#888}{x_w^T}\color{#88bd33}{u_{cat}}}\color{white})).$
    
Negative sampling is efficient because a word only has few "true" contexts. So randomly chosen words are likely to be negative only. Negative samples are randomly selected over the probability distribution.

#### Variants of Word2Vec
- Skip-gram: predicts context words from central word.
- CBOW (Continuous Bag of Words): predicts central word from sum of context vectors. 

### GloVe: Global Vectors for Word Representation
- combined count and prediction methods
- Use global information from corpus to learn vectors
- Simple co-occurence used to measure (w, c) association. Loss formula:

- ![alt text](https://github.com/shollercoaster/AI-Notes-And-Resources/blob/main/images/glove-loss-function.png "GloVe Loss Formula")

Same learnt params: context and central word vectors. Also has scalar bias for each.

Glove controls influence of rare and frequent words:
- rare events penalized (less weight)
- Frequent events are not overweighted

## Evaluation of Word Embeddings

#### Intrinsic evaluation
- how well vectors capture meaning
- many semantic and syntactic word relationships are linear (king-queen, man-woman)
- Can also find linear relationships BETWEEN semantic spaces (that is corresponding words in 2 different languages will match in the new, joint semantic space).

#### Extrinsic evaluation
- how well they perform on a specific task - text classification etc
