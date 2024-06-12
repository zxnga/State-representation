import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.buffers import ReplayBuffer
from sklearn.model_selection import train_test_split


def create_replay_buffer(env, building_count, size):
    buffer = ReplayBuffer(buffer_size=size*building_count, 
                          observation_space=env.observation_space[0],
                          action_space=env.action_space[0],
                          handle_timeout_termination=False)
    return buffer

def populate_replay_buffer(env, buffer, size, n_agent):
    print('populating replay buffer...')
    obs, info = env.reset()
    for _ in range(size):
        actions = get_action(n_agent)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        #assuming we want to store each agent's info singularly
        for i in range(n_agent):
            buffer.add(np.array(obs[i]), np.array(next_obs[i]), np.array(actions[i]), np.array(reward[i]), np.array(terminated), np.array(info))
        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()

def get_action(n_agent, low=-1, high=1):
    return [[random.uniform(low, high)] for _ in range(n_agent)]

def extract_data_from_buffer(buffer, input_dim, action_dim, test_size, validation_size):
    # Extract and reshape data from buffer
    states = buffer.observations.reshape(-1, input_dim)
    actions = buffer.actions.reshape(-1, action_dim)
    next_states = buffer.next_observations.reshape(-1, input_dim)
    rewards = buffer.rewards.reshape(-1, 1)  # Assuming rewards are single-dimensional

    # Split data into training and test sets
    (train_states, test_states, train_actions, test_actions,
    train_next_states, test_next_states, train_rewards, test_rewards) = train_test_split(
        states, actions, next_states, rewards, test_size=test_size, random_state=42)

    # Further split the training data into training and validation sets
    (train_states, val_states, train_actions, val_actions,
    train_next_states, val_next_states, train_rewards, val_rewards) = train_test_split(
        train_states, train_actions, train_next_states, train_rewards, test_size=validation_size, random_state=42)

    return (train_states, val_states, test_states, train_actions, val_actions,
            test_actions, train_next_states, val_next_states, test_next_states,
            train_rewards, val_rewards, test_rewards)


def prepare_dataloaders(train_states, val_states, test_states, train_actions, 
                        val_actions, test_actions, train_next_states, val_next_states,
                        test_next_states, train_rewards, val_rewards, test_rewards,
                        batch_size):
    train_dataset = TensorDataset(torch.tensor(train_states, dtype=torch.float32), 
                                  torch.tensor(train_actions, dtype=torch.float32), 
                                  torch.tensor(train_next_states, dtype=torch.float32),
                                  torch.tensor(train_rewards, dtype=torch.float32))
    
    val_dataset = TensorDataset(torch.tensor(val_states, dtype=torch.float32), 
                                torch.tensor(val_actions, dtype=torch.float32), 
                                torch.tensor(val_next_states, dtype=torch.float32),
                                torch.tensor(val_rewards, dtype=torch.float32))
    
    test_dataset = TensorDataset(torch.tensor(test_states, dtype=torch.float32), 
                                 torch.tensor(test_actions, dtype=torch.float32), 
                                 torch.tensor(test_next_states, dtype=torch.float32),
                                 torch.tensor(test_rewards, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_combined_model(combined_model, state_dim, train_loader, val_loader, test_loader, 
                         num_epochs, reconstruction_criterion, prediction_criterion, 
                         inverse_criterion, reward_criterion, optimizer, plot_dir, weights_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Lists to store the losses
    train_reconstruction_losses = []
    train_prediction_losses = []
    train_inverse_losses = []
    train_reward_losses = []
    val_reconstruction_losses = []
    val_prediction_losses = []
    val_inverse_losses = []
    val_reward_losses = []

    # Training loop
    for epoch in range(num_epochs):
        combined_model.train()
        epoch_train_reconstruction_loss = 0.0
        epoch_train_prediction_loss = 0.0
        epoch_train_inverse_loss = 0.0
        epoch_train_reward_loss = 0.0

        for states, actions, next_states, rewards in train_loader:
            states, actions, next_states, rewards = states.to(device), actions.to(device), next_states.to(device), rewards.to(device)

            # Forward pass
            outputs = combined_model(states, actions)
            decoded_state, decoded_predicted_next_state, encoded_state, predicted_encoded_next_state, predicted_action, predicted_reward = outputs

            # Encode the actual next state
            encoded_next_state, _ = combined_model.autoencoder(next_states) if combined_model.forward_model is not None else (None, None)

            # Compute losses
            reconstruction_loss = reconstruction_criterion(decoded_state, states)
            prediction_loss = prediction_criterion(predicted_encoded_next_state, encoded_next_state) if combined_model.forward_model is not None else 0.0
            inverse_loss = inverse_criterion(predicted_action, actions) if combined_model.inverse_model is not None else 0.0
            reward_loss = reward_criterion(predicted_reward, rewards) if combined_model.reward_model is not None else 0.0

            # Combine losses
            loss = reconstruction_loss + prediction_loss + inverse_loss + reward_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_reconstruction_loss += reconstruction_loss.item() * states.size(0)
            epoch_train_prediction_loss += prediction_loss.item() * states.size(0) if combined_model.forward_model is not None else 0.0
            epoch_train_inverse_loss += inverse_loss.item() * states.size(0) if combined_model.inverse_model is not None else 0.0
            epoch_train_reward_loss += reward_loss.item() * states.size(0) if combined_model.reward_model is not None else 0.0

        epoch_train_reconstruction_loss /= len(train_loader.dataset)
        epoch_train_prediction_loss /= len(train_loader.dataset)
        epoch_train_inverse_loss /= len(train_loader.dataset)
        epoch_train_reward_loss /= len(train_loader.dataset)
        train_reconstruction_losses.append(epoch_train_reconstruction_loss)
        train_prediction_losses.append(epoch_train_prediction_loss)
        train_inverse_losses.append(epoch_train_inverse_loss)
        train_reward_losses.append(epoch_train_reward_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Reconstruction Loss: {epoch_train_reconstruction_loss:.4f}, '
              f'Training Prediction Loss: {epoch_train_prediction_loss:.4f}, Training Inverse Loss: {epoch_train_inverse_loss:.4f}, '
              f'Training Reward Loss: {epoch_train_reward_loss:.4f}')

        # Validation
        combined_model.eval()
        epoch_val_reconstruction_loss = 0.0
        epoch_val_prediction_loss = 0.0
        epoch_val_inverse_loss = 0.0
        epoch_val_reward_loss = 0.0
        with torch.no_grad():
            for states, actions, next_states, rewards in val_loader:
                states, actions, next_states, rewards = states.to(device), actions.to(device), next_states.to(device), rewards.to(device)
                outputs = combined_model(states, actions)
                decoded_state, decoded_predicted_next_state, encoded_state, predicted_encoded_next_state, predicted_action, predicted_reward = outputs

                # Encode the actual next state
                encoded_next_state, _ = combined_model.autoencoder(next_states) if combined_model.forward_model is not None else (None, None)

                reconstruction_loss = reconstruction_criterion(decoded_state, states)
                prediction_loss = prediction_criterion(predicted_encoded_next_state, encoded_next_state) if combined_model.forward_model is not None else 0.0
                inverse_loss = inverse_criterion(predicted_action, actions) if combined_model.inverse_model is not None else 0.0
                reward_loss = reward_criterion(predicted_reward, rewards) if combined_model.reward_model is not None else 0.0

                epoch_val_reconstruction_loss += reconstruction_loss.item() * states.size(0)
                epoch_val_prediction_loss += prediction_loss.item() * states.size(0) if combined_model.forward_model is not None else 0.0
                epoch_val_inverse_loss += inverse_loss.item() * states.size(0) if combined_model.inverse_model is not None else 0.0
                epoch_val_reward_loss += reward_loss.item() * states.size(0) if combined_model.reward_model is not None else 0.0

        epoch_val_reconstruction_loss /= len(val_loader.dataset)
        epoch_val_prediction_loss /= len(val_loader.dataset)
        epoch_val_inverse_loss /= len(val_loader.dataset)
        epoch_val_reward_loss /= len(val_loader.dataset)
        val_reconstruction_losses.append(epoch_val_reconstruction_loss)
        val_prediction_losses.append(epoch_val_prediction_loss)
        val_inverse_losses.append(epoch_val_inverse_loss)
        val_reward_losses.append(epoch_val_reward_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Reconstruction Loss: {epoch_val_reconstruction_loss:.4f}, '
              f'Validation Prediction Loss: {epoch_val_prediction_loss:.4f}, Validation Inverse Loss: {epoch_val_inverse_loss:.4f}, '
              f'Validation Reward Loss: {epoch_val_reward_loss:.4f}')

    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_reconstruction_losses, label='Training Reconstruction Loss')
    plt.plot(range(1, num_epochs + 1), train_prediction_losses, label='Training Prediction Loss')
    plt.plot(range(1, num_epochs + 1), train_inverse_losses, label='Training Inverse Loss')
    plt.plot(range(1, num_epochs + 1), train_reward_losses, label='Training Reward Loss')
    plt.plot(range(1, num_epochs + 1), val_reconstruction_losses, label='Validation Reconstruction Loss')
    plt.plot(range(1, num_epochs + 1), val_prediction_losses, label='Validation Prediction Loss')
    plt.plot(range(1, num_epochs + 1), val_inverse_losses, label='Validation Inverse Loss')
    plt.plot(range(1, num_epochs + 1), val_reward_losses, label='Validation Reward Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()

    plt.savefig(os.path.join(plot_dir, f'CombinedModel_{combined_model.type_ae}_training_validation_losses_{state_dim}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Evaluation on the test set
    combined_model.eval()
    with torch.no_grad():
        total_reconstruction_loss = 0.0
        total_prediction_loss = 0.0
        total_inverse_loss = 0.0
        total_reward_loss = 0.0
        for states, actions, next_states, rewards in test_loader:
            states, actions, next_states, rewards = states.to(device), actions.to(device), next_states.to(device), rewards.to(device)
            outputs = combined_model(states, actions)
            decoded_state, decoded_predicted_next_state, encoded_state, predicted_encoded_next_state, predicted_action, predicted_reward = outputs

            # Encode the actual next state
            encoded_next_state, _ = combined_model.autoencoder(next_states) if combined_model.forward_model is not None else (None, None)

            reconstruction_loss = reconstruction_criterion(decoded_state, states)
            prediction_loss = prediction_criterion(predicted_encoded_next_state, encoded_next_state) if combined_model.forward_model is not None else 0.0
            inverse_loss = inverse_criterion(predicted_action, actions) if combined_model.inverse_model is not None else 0.0
            reward_loss = reward_criterion(predicted_reward, rewards) if combined_model.reward_model is not None else 0.0

            total_reconstruction_loss += reconstruction_loss.item() * states.size(0)
            total_prediction_loss += prediction_loss.item() * states.size(0) if combined_model.forward_model is not None else 0.0
            total_inverse_loss += inverse_loss.item() * states.size(0) if combined_model.inverse_model is not None else 0.0
            total_reward_loss += reward_loss.item() * states.size(0) if combined_model.reward_model is not None else 0.0

        total_reconstruction_loss /= len(test_loader.dataset)
        total_prediction_loss /= len(test_loader.dataset)
        total_inverse_loss /= len(test_loader.dataset)
        total_reward_loss /= len(test_loader.dataset)
        print(f'Test Reconstruction Loss: {total_reconstruction_loss:.4f}')
        print(f'Test Prediction Loss: {total_prediction_loss:.4f}')
        print(f'Test Inverse Loss: {total_inverse_loss:.4f}')
        print(f'Test Reward Loss: {total_reward_loss:.4f}')

    # Save the models
    save_dir = os.path.join(weights_dir, f'{type_ae}/state_dim_{state_dim}')
    combined_model.save(save_dir)
    combined_model.save(save_dir, save_whole_model=False)