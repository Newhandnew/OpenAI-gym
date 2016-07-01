import gym
import tensorflow as tf
import numpy as numpy
import random
from collections import deque

# Hyper Parameters for DQN
Gamma = 0.9  			# discount facor for target Q
Iintial_epsilon = 0.5	# starting value of epsilon
Final_epsilon = 0.01	# final value of epsilon
Replay_size = 10000		# experience replay buffer size
Batch_size = 32			# size of minibatch
Episode = 10000 		# Episode limitation

class DQN():
	# DQN Agent
	def __init__(self, env):
		# initial experience replay
		self.replay_buffer = deque()
		self.time_step = 0
		self.epsilon = Iintial_epsilon
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n

		self.create_Q_network()
		self.creat_training_method()

		self.session = tf.InteractiveSession()
		self.session.run(tf..initialize_all_variables())

		#loading networks
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.session, checkpoint.model_checkpoint_path)
			print "Successfully loaded", model_checkpoint_path
		else:
			print "Can't find saved network weights"
		self.summary_writer =  tf.train.SummaryWriter('log', graph=self.session.graph)

	def create_Q_network(self):
		# network weights
		num_hidden_node = 30
		w1 = self.weight_variable([self.state_dim, num_hidden_node])
		b1 = self.bias_variable([num_hidden_node])
		w2 = self.weight_variable([num_hidden_node, self.action_dim])
		b2 = self.bias_variable([self.action_dim])
		# input layer
		self.state_input = tf.placeholder("float", [None, self.state_dim])
		# hidden layers
		h_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
		# Q value layer
		self.Q_value = tf.matmul(h_layer, w2) + b2

	def create_training_method(self):
		self.action_input = tf.placeholder("float", [None, self.action_dim])	# one hot presentation
		self.y_input = tf.placeholder("float", [None])
		Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		# summary
		tf.scalar_summary("loss", self.cost)
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

	def perceive(self, state, action, reward, next_state, done):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		self.replay_buffer.append(state, one_hot_action, reward, next_state, done)
		if len(self.replay_buffer) > Replay_size:
			self.replay_buffer.popleft()
		if len(self.replay_buffer) > Batch_size:
			self.train_Q_network()

	def train_Q_network(self):
		self.time_step += 1
		# Step 1: obtain random minibatch from replay memory
		minbatch = random.sample(self.replay_buffer, Batch_size)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]

		# Step 2: calculate y
		y_batch = []
		Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
		for i in xrange(Batch_size):
			done = minibatch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + Gamma * np.max(Q_value_batch[i]))

		self.optimizer.run(feed_dict={
			self.y_input:y_batch,
			self.action_input:action_batch,
			self.state_input:state_batch
			})
		summary_str = self.session.run(merged_summary_op, feed_dict={
			self.y_input:y_batch,
			self.action_input:action_batch,
			self.state_input:state_batch
			})
		self.summary_writer.add_summary(summary_str, self.time_step)

		# save network every 1000 iteration
		if self.time_step % 1000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.time_step)

	def egreedy_action(self, state):
		Q_value = self.Q_value.eval(feed_dict={
			self.state_input:[state]
			})[0]
		if random.random() <= self.epsilon:
			return random.randint(0, self.action_dim - 1)
		else:
			return np.argmax(Q_value)

		self.epsilon -= (Iintial_epsilon - Final_epsilon) / Episode

	def action(self, state):
		return np.argmax(self.Q_value.eval(feed_dict={
			self.state_input:[state]
			})[0])

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

# -------------------------------------------------------------------

ENV_NAME = 'CartPole-v0'
STEP = 300				# step limit in an episode
TEST = 10				# number of experiment test every 100 episode


def main():
	# initialize OpenAI Gym env and dqn agent
	env = gym.make(ENV_NAME)
	agent = DQN(env)

	for episode in xrange(EPISODE):
		# initialize task
		state = env.reset()
		# Train 
		for step in xrange(STEP):
			action = agent.egreedy_action(state) # e-greedy action for train
			next_state,reward,done,_ = env.step(action)
			agent.perceive(state,action,reward,next_state,done)
			state = next_state
			if done:
				break
		# Test every 100 episodes
		if episode % 100 == 0:
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(STEP):
					env.render()
					action = agent.action(state) # direct action for test
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST
			print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
			if ave_reward >= 200:
				break

	# save results for uploading
	env.monitor.start('gym_results/CartPole-v0-experiment-1',force = True)
	for i in xrange(100):
		state = env.reset()
		for j in xrange(200):
			env.render()
			action = agent.action(state) # direct action for test
			state,reward,done,_ = env.step(action)
			total_reward += reward
			if done:
				break
	env.monitor.close()

if __name__ == '__main__':
	main()