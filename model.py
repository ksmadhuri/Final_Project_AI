import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import os

class QNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_tensor):
        activation = functional.relu(self.layer1(input_tensor))
        output = self.layer2(activation)
        return output

    def save_model(self, filename='model.pth'):
        directory_path = './model'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        filepath = os.path.join(directory_path, filename)
        torch.save(self.state_dict(), filepath)


class QLearningTrainer:
    def __init__(self, model, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def update(self, current_state, chosen_action, reward_received, next_state, terminal):
        current_state = torch.tensor(current_state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        chosen_action = torch.tensor(chosen_action, dtype=torch.long)
        reward_received = torch.tensor(reward_received, dtype=torch.float)
        
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            chosen_action = chosen_action.unsqueeze(0)
            reward_received = reward_received.unsqueeze(0)
            terminal = (terminal, )

        predicted_q_values = self.model(current_state)

        q_targets = predicted_q_values.clone()
        for idx in range(len(terminal)):
            updated_q_value = reward_received[idx]
            if not terminal[idx]:
                updated_q_value = reward_received[idx] + self.discount_factor * torch.max(self.model(next_state[idx]))

            q_targets[idx][torch.argmax(chosen_action[idx]).item()] = updated_q_value

        self.optimizer.zero_grad()
        loss = self.loss_function(q_targets, predicted_q_values)
        loss.backward()
        self.optimizer.step()
