State Representation for RL algorithms:

- Autoencoder based learning representations using auxiliary objectives to learn a meaningful representation
- Objectives:
    - Forward: predict next state given current state and action
    - Inverse: predict action given current state and next state
    - Reward: predict reward given current state and next state

- Training is done using a dataset of expriences collected in a ReplyBuffer
- Once the model is trained we can discard the auxiliary models to only use the trained encoder to extract a state representation
- We can also use the trainde models in a boostraping method to evaluate future states and actions