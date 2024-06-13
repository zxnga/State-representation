State Representation for RL algorithms:

- Autoencoder based representation learning using auxiliary objectives to learn meaningful representations
- Objectives:
    - Reconstruction: base objective of the Autoencoder, predict a state given its latent representation
    - Forward: predict next state given current state and action
    - Inverse: predict action given current state and next state
    - Reward: predict reward given current state and next state
- Objectives can be coupled in every way but the encoder learns a single representation mixing every objective, to learn a unique representation for every objective one must create an AE instance for each objective

- Training is done using a dataset of expriences collected in a ReplayBuffer
- Once the model is trained we can discard the auxiliary models to only use the trained encoder to extract a state representation
- We can also use the trained models in a boostraping method to evaluate future states and actions