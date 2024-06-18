import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium
import numpy as np
import pandas as pd
import math
import random
import os

# widely built upon https://github.com/araffin/srl-zoo/tree/master

class BaseModelSRL(nn.Module):
    """
    Base Class for a SRL network
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self):
        super(BaseModelSRL, self).__init__()

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.forward(observations)

    def forward(self, x):
        raise NotImplementedError


class BaseModelAutoEncoder(BaseModelSRL):
    """
    Base Class for a SRL network (autoencoder family)
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self):
        super(BaseModelAutoEncoder, self).__init__()

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encode(observations)

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def forward(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        encoded = self.encode(x)
        decoded = self.decode(encoded).view(input_shape)
        return encoded, decoded

class DenseAutoEncoder(BaseModelAutoEncoder):
    """
    Dense autoencoder network
    Known issue: it reconstructs the image but omits the robot arm
    :param input_dim: (int)
    :param state_dim: (int)
    """

    def __init__(self, input_dim, output_dim, state_dim=3):
        super(DenseAutoEncoder, self).__init__()

        self.name = 'DenseAE'
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, output_dim),
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        # Flatten input
        # x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.decoder(x)

class BaseForwardModel(BaseModelSRL):
    def __init__(self):
        self.action_dim = None
        self.forward_net = None
        self.name = 'ForwardModel'
        super(BaseForwardModel, self).__init__()

    def initForwardNet(self, state_dim, action_dim, n_hidden=50, model_type='mlp'):
        self.action_dim = action_dim
        if model_type == "linear":
            self.forward_net = nn.Linear(state_dim + action_dim, state_dim)
        elif model_type == "mlp":
            self.forward_net = nn.Sequential(nn.Linear(state_dim + action_dim, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, state_dim)
                                             )
        else:
            raise ValueError("Unknown model_type for inverse model: {}".format(model_type))

    def forward(self, x):
        raise NotImplementedError()

    # def forwardModel(self, state, action):
    #     """
    #     Predict next state given current state and action
    #     :param state: (th.Tensor)
    #     :param action: (th Tensor)
    #     :return: (th.Tensor)
    #     """
    #     # Predict the delta between the next state and current state
    #     # by taking as input concatenation of state & action over the 2nd dimension
    #     concat = torch.cat((state, encodeOneHot(action, self.action_dim)), dim=1)
    #     return state + self.forward_net(concat)

    def forwardModel(self, state, action):
        """
            Predict next state given current state and action
            :param state: (torch.Tensor)
            :param action: (torch.Tensor)
            :return: (torch.Tensor)
        """
        return self.forward_net(torch.cat((state, action), dim=1))

# class ForwardModel(BaseForwardModel):
#     def __init__(self, state_dim, action_dim):
#         super(ForwardModel, self).__init__()
#         self.name = 'ForwardModel'
#         self.initForwardNet(state_dim, action_dim)

#     def forward(self, state, action):
#         return self.forwardModel(state, action)


class BaseInverseModel(BaseModelSRL):
    def __init__(self):
        self.inverse_net = None
        self.name = 'InverseModel'
        super(BaseInverseModel, self).__init__()

    def initInverseNet(self, state_dim, action_dim, n_hidden=128, model_type="linear"):
        """
        :param state_dim: (th.Tensor)
        :param action_dim: (int)
        :param n_hidden: (int)
        :param model_type: (str)
        :return: (th.Tensor)
        """
        if model_type == "linear":
            self.inverse_net = nn.Linear(state_dim * 2, action_dim)
        elif model_type == "mlp":
            self.inverse_net = nn.Sequential(nn.Linear(state_dim * 2, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, action_dim)
                                             )
        else:
            raise ValueError("Unknown model_type for inverse model: {}".format(model_type))

    def forward(self, x):
        raise NotImplementedError()

    def inverseModel(self, state, next_state):
        """
        Predict action given current state and next state
        :param state: (th.Tensor)
        :param next_state: (th.Tensor)
        :return: probability of each action
        """
        # input: concatenation of state & next state over the 2nd dimension
        return self.inverse_net(torch.cat((state, next_state), dim=1))


class BaseRewardModel(BaseModelSRL):
    def __init__(self):
        self.reward_net = None
        self.name = 'RewardModel'
        super(BaseRewardModel, self).__init__()

    def initRewardNet(self, state_dim, n_rewards=1, n_hidden=16, model_type="mlp"):
        if model_type == "linear":
            self.reward_net = nn.Linear(state_dim * 2, n_rewards)
        elif model_type == "mlp":
            self.reward_net = nn.Sequential(nn.Linear(state_dim * 2, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, n_rewards)
                                             )
        else:
            raise ValueError("Unknown model_type for inverse model: {}".format(model_type))
    def forward(self, x):
        raise NotImplementedError()

    def rewardModel(self, state, next_state):
        """
        Predict reward given current state and next state
        :param state: (th.Tensor)
        :param next_state: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.reward_net(torch.cat((state, next_state), dim=1))

class CombinedModel(nn.Module):
    def __init__(self, autoencoder, forward_model=None, inverse_model=None, reward_model=None):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.reward_model = reward_model
        self.name = "combinedAE"
        self.type_ae = ''
        if self.forward_model is not None:
            self.name += "Forward"
            self.type_ae += 'forward'
        if self.inverse_model is not None:
            self.name += "Inverse"
            self.type_ae += 'inverse'
        if self.reward_model is not None:
            self.name += "Reward"
            self.type_ae += 'reward'
        if self.type_ae == '':
            self.type_ae = 'base'

    # def forward(self, state, action):
    #     # Encode the state
    #     encoded_state, decoded_state = self.autoencoder(state)
        
    #     if self.forward_model is not None:
    #         # Predict the next encoded state
    #         predicted_encoded_next_state = self.forward_model.forwardModel(encoded_state, action)
    #         # Decode the predicted encoded state
    #         decoded_predicted_next_state = self.autoencoder.decoder(predicted_encoded_next_state)
    #     else:
    #         predicted_encoded_next_state = None
    #         decoded_predicted_next_state = None
        
    #     if self.inverse_model is not None:
    #         # Predict the action from the inverse model
    #         if predicted_encoded_next_state is not None:
    #             predicted_action = self.inverse_model.inverseModel(encoded_state, predicted_encoded_next_state)
    #         else:
    #             predicted_action = self.inverse_model.inverseModel(encoded_state, encoded_state)
    #     else:
    #         predicted_action = None
        
    #     if self.reward_model is not None and predicted_encoded_next_state is not None:
    #         predicted_reward = self.reward_model.rewardModel(encoded_state, predicted_encoded_next_state)
    #     else:
    #         predicted_reward = None
        
    #     return decoded_state, decoded_predicted_next_state, encoded_state, predicted_encoded_next_state, predicted_action, predicted_reward

    def forward(self, state, action, next_state=None):
        # Encode the state
        encoded_state, decoded_state = self.autoencoder(state)
        
        if self.forward_model is not None:
            # Predict the next encoded state
            predicted_encoded_next_state = self.forward_model.forwardModel(encoded_state, action)
            # Decode the predicted encoded state
            decoded_predicted_next_state = self.autoencoder.decoder(predicted_encoded_next_state)
        else:
            predicted_encoded_next_state = None
            decoded_predicted_next_state = None

        if self.inverse_model is not None:
            if next_state is not None:
                # Encode the next state
                encoded_next_state, _ = self.autoencoder(next_state)
            else:
                encoded_next_state = predicted_encoded_next_state
        
            # Predict the action from the inverse model
            if encoded_next_state is not None:
                predicted_action = self.inverse_model.inverseModel(encoded_state, encoded_next_state)
            else:
                predicted_action = None
        else:
            predicted_action = None

        if self.reward_model is not None:
            if next_state is not None:
                # Encode the next state
                encoded_next_state, _ = self.autoencoder(next_state)
            else:
                encoded_next_state = predicted_encoded_next_state

            if encoded_next_state is not None:
                predicted_reward = self.reward_model.rewardModel(encoded_state, encoded_next_state)
            else:
                predicted_reward = None
        else:
            predicted_reward = None

        return decoded_state, decoded_predicted_next_state, encoded_state, predicted_encoded_next_state, predicted_action, predicted_reward

    def save(self, save_dir, save_whole_model=True):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_whole_model:
            n_models = len([i for i in os.listdir(save_dir) if i.startswith(f'{self.name}_dim{self.autoencoder.state_dim}_')])
            save_path = os.path.join(save_dir, f'{self.name}_dim{self.autoencoder.state_dim}_{n_models}.pth')
            torch.save(self.state_dict(), save_path)
            print(f'Whole model saved to {save_path}')
        else:
            n_models = len([i for i in os.listdir(save_dir) if i.startswith(f'{self.autoencoder.name}_encoder_dim{self.autoencoder.state_dim}_')])
            encoder_path = os.path.join(save_dir, f'{self.autoencoder.name}_encoder_dim{self.autoencoder.state_dim}_{n_models}.pth')
            torch.save(self.autoencoder.state_dict(), encoder_path)
            print(f'Encoder saved to {encoder_path}')
            
            if self.forward_model is not None:
                n_models = len([i for i in os.listdir(save_dir) if i.startswith(f'{self.forward_model.name}_')])
                forward_model_path = os.path.join(save_dir, f'{self.forward_model.name}_{n_models}.pth')
                torch.save(self.forward_model.state_dict(), forward_model_path)
                print(f'Forward model saved to {forward_model_path}')
            
            if self.inverse_model is not None:
                n_models = len([i for i in os.listdir(save_dir) if i.startswith(f'{self.inverse_model.name}_')])
                inverse_model_path = os.path.join(save_dir, f'{self.inverse_model.name}_encoder_dim{self.autoencoder.state_dim}_{n_models}.pth')
                torch.save(self.inverse_model.state_dict(), inverse_model_path)
                print(f'Inverse model saved to {inverse_model_path}')
            
            if self.reward_model is not None:
                n_models = len([i for i in os.listdir(save_dir) if i.startswith(f'{self.reward_model.name}_')])
                reward_model_path = os.path.join(save_dir, f'{self.reward_model.name}_{n_models}.pth')
                torch.save(self.reward_model.state_dict(), reward_model_path)
                print(f'Reward model saved to {reward_model_path}')

    def load(self, load_path, load_whole_model=True):
        if load_whole_model:
            self.load_state_dict(torch.load(load_path))
            print(f'Whole model loaded from {load_path}')
        else:
            encoder_path = load_path['encoder']
            self.autoencoder.load_state_dict(torch.load(encoder_path))
            print(f'Encoder loaded from {encoder_path}')
            
            if 'forward_model' in load_path and self.forward_model is not None:
                forward_model_path = load_path['forward_model']
                self.forward_model.load_state_dict(torch.load(forward_model_path))
                print(f'Forward model loaded from {forward_model_path}')
            
            if 'inverse_model' in load_path and self.inverse_model is not None:
                inverse_model_path = load_path['inverse_model']
                self.inverse_model.load_state_dict(torch.load(inverse_model_path))
                print(f'Inverse model loaded from {inverse_model_path}')
            
            if 'reward_model' in load_path and self.reward_model is not None:
                reward_model_path = load_path['reward_model']
                self.reward_model.load_state_dict(torch.load(reward_model_path))
                print(f'Reward model loaded from {reward_model_path}')
