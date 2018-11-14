# Hyperparameters, Swarm Intelligence

We have used several hyperparameters in both training and testing of our model with ANN, such as number of input neurons, number of hidden layers, output of hidden neurons, activation functions, batch_size, number of epochs and etc. In order to get high accuracy, we have to optimize these variables. In parameter tuning section, we saw how to use GridSearch technique to train model multiple times and chose best one. Another method to increase accuracy is decent and innovative approach which is called Partical Swarm Optimization. PSO is one of the Swarm Intelligence technique to mimic insects - such as bee, fish, ant and turn their natural social behaviours to algorithm. In real life, bees do Waggle dance in order to share area of flowers to others, ants build nests with colony in parallel, birds fly in flocks and etc. Nowadays, scientists try to learn their behaviour and apply them in mathematical problems. For example, Ant Colony Optimization was used to solve Travelling Salesman Problem. For ANN perspective, in some occasion, Gradient Descent is stucking in local minimum, in that case SI can be ideal to find global one. Table 1 illustrates the multiple SI algorithms.

<img width="633" alt="screen shot 2018-11-14 at 10 15 31 am" src="https://user-images.githubusercontent.com/5506152/48467335-05d19500-e802-11e8-887a-ff9695398081.png">

# PSO
• Introduced by Kennedy & Eberhart 1995
• Inspired by social behavior and movement
dynamics of insects, birds and fish
• Global gradient-less stochastic search method
• Suited to continuous variable problems
• Performance comparable to Genetic algorithms
• Has successfully been applied to a wide variety
of problems (Neural Networks, Structural opt.,
Shape topology opt.)

### PSO applications
• Training of neural networks
  – Identification of Parkinson’s disease
  – Extraction of rules from fuzzy networks
  – Image recognition
• Optimization of electric power distribution
networks
• Structural optimization
  – Optimal shape and sizing design
  – Topology optimization
• Process biochemistry
• System identification in biomechanics

### Advantages and Disadvantages
• Advantages
  – Insensitive to scaling of design variables
  – Simple implementation
  – Easily parallelized for concurrent processing
  – Derivative free
  – Very few algorithm parameters
  – Very efficient global search algorithm
• Disadvantages
  – Slow convergence in refined search stage (weak
  local search ability)

<img width="1161" alt="screen shot 2018-11-14 at 11 24 29 am" src="https://user-images.githubusercontent.com/5506152/48467342-066a2b80-e802-11e8-8e04-35a6a07578ba.png">

<img width="1142" alt="screen shot 2018-11-14 at 11 24 51 am" src="https://user-images.githubusercontent.com/5506152/48467344-066a2b80-e802-11e8-8a23-f02f586c4c26.png">

<img width="961" alt="screen shot 2018-11-14 at 11 25 01 am" src="https://user-images.githubusercontent.com/5506152/48467345-066a2b80-e802-11e8-9cd6-d5c651f8623e.png">

<img width="974" alt="screen shot 2018-11-14 at 11 24 11 am" src="https://user-images.githubusercontent.com/5506152/48467341-066a2b80-e802-11e8-9afa-aad41ffbc246.png">

<img width="1246" alt="screen shot 2018-11-14 at 11 23 29 am" src="https://user-images.githubusercontent.com/5506152/48467338-05d19500-e802-11e8-9481-f1bac9b9b952.png">

<img width="1156" alt="screen shot 2018-11-14 at 11 21 19 am" src="https://user-images.githubusercontent.com/5506152/48467337-05d19500-e802-11e8-9e3f-23572df6f362.png">

<img width="1048" alt="screen shot 2018-11-14 at 11 21 00 am" src="https://user-images.githubusercontent.com/5506152/48467336-05d19500-e802-11e8-8af7-30a9b978215e.png">


References:
1. https://en.wikipedia.org/wiki/Swarm_intelligence
2. https://www.youtube.com/watch?v=JhgDMAm-imI 
3. https://www.youtube.com/watch?v=LU_KD1enR3Q
4. Schutte, Jaco F. "The Particle Swarm Optimization Algorithm." Structural Optimization (2005).
