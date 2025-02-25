import gymnasium as gym
import numpy as np
from QLearning import QLearning
import matplotlib.pylab as plt

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

state_space = env.observation_space.n
action_space = env.action_space.n
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1
epsilon_decay = 0.99995
epsilon_min = 0.001
max_steps = 1000
num_episodes = 100000

model = QLearning(
    state_space=state_space,
    action_space=action_space,
    learning_rate=learning_rate,
    discount_factor=discount_factor,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_min=epsilon_min
)

rewards = []
epochs = []
td_errors = []
epsilons = []
total_reward = 0
for ep in range(num_episodes):
    state, info = env.reset()
    model.epsilon_update()
    ep_loss = 0
    for step in range(max_steps):
        action = model.predict(state)
        next_state, reward, _, done, info = env.step(action)
        loss = model.backward(state, reward, next_state)
        total_reward += reward
        ep_loss += loss
        state = next_state
        if done:
            break
    epsilons.append(model.epsilon)
    td_errors.append(ep_loss)
    epochs.append(step)
    rewards.append(total_reward)



fg, axis = plt.subplots(3, 1, layout='constrained')

axis[0].plot(range(num_episodes), rewards)
axis[0].set_xlabel("number of episodes")
axis[0].set_ylabel("Sum of rewards")

axis[1].plot(range(num_episodes), td_errors)
axis[1].set_xlabel("number of episodes")
axis[1].set_ylabel("TD Error")

axis[2].plot(range(num_episodes), epsilons)
axis[2].set_xlabel("number of episodes")
axis[2].set_ylabel("epsilon")

plt.show()