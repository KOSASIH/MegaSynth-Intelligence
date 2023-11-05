import gym
import numpy as np
import tensorflow as tf

# Define the policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(action_size, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

# Define the agent
class ReinforcementLearningAgent:
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam()

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        action_probs = self.policy_network(state)
        return np.random.choice(self.action_size, p=action_probs.numpy()[0])

    def update_policy(self, states, actions, rewards):
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(states)
            selected_action_probs = tf.reduce_sum(action_probs * tf.one_hot(actions, self.action_size), axis=1)
            loss = -tf.reduce_mean(tf.math.log(selected_action_probs) * rewards)

        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

    def train(self, num_episodes, max_steps, epsilon):
        rewards_history = []
        episode_lengths = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0

            for step in range(max_steps):
                if np.random.rand() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_action(state)

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    reward = -10

                self.update_policy(np.reshape(state, [1, self.state_size]),
                                   np.array([action]),
                                   np.array([reward]))

                state = next_state

                if done or step == max_steps - 1:
                    rewards_history.append(total_reward)
                    episode_lengths.append(step + 1)
                    print("Episode:", episode+1, "Total Reward:", total_reward)
                    break

        return rewards_history, episode_lengths

# Create the environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create the agent
agent = ReinforcementLearningAgent(env, state_size, action_size)

# Train the agent
num_episodes = 1000
max_steps = 500
epsilon = 0.2
rewards_history, episode_lengths = agent.train(num_episodes, max_steps, epsilon)

# Markdown code output showcasing the agent's performance metrics
print("Average Reward:", np.mean(rewards_history))
print("Average Episode Length:", np.mean(episode_lengths))
