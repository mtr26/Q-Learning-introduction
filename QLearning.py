import numpy as np

class QLearning:
    def __init__(self, state_space, action_space, learning_rate=1e-3, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.last_action_ = None
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initialize_qtable()
    
    def initialize_qtable(self):
        self.qtable = np.zeros((self.state_space, self.action_space))

    def predict(self, state):
        if np.random.random() > self.epsilon:
            self.last_action_ = np.argmax(self.qtable[state])
        else:
            self.last_action_ = np.random.randint(0, self.action_space)
        return self.last_action_
    
    def epsilon_update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def backward(self, state, reward, next_state):
        td_error = reward + self.discount_factor * np.max(self.qtable[next_state]) - self.qtable[state][self.last_action_]
        self.qtable[state][self.last_action_] += self.lr * td_error
        return td_error
