# dqn-perudo

Hi,this is the initial version of using DQN to play Perudo against a robot. 

There are 4 layers that had been created. 
 
The first layer takes a state-action pair as an input layer. The middle layers consist of two hidden layers. The last layer is a Q-value outputlayer, used to conduct the next step. 
 
The action space contains ‘Dudo’ and all possible valid bids.
 
To connect each layer, l use an activation function, Leaky-Relu in Pytorch(since it has a faster convergence speed and lower computational complexity compared with alternative activation functions), to map the input layer to the hidden layers.

The memory set can be changed from a low value to high value, l found that with around 1,000 memory samples is not enough for its low winning rate.

However, for higher memory around 100,000, it will increase the winning rate to around 75%, althought it is not as high as methods like Q-Learning(around 85%).

To continue, l increase the memory to a much higher level but it didn't give much improved, maybe the winning rate can be increased by change the hidden layers.

Anyways, feel free to use my code for any types of further research!!
