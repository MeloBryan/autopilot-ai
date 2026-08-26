import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class AirplaneEnv:
	def __init__(self, altitude_cible=3000.0):
		self.altitude_cible = altitude_cible
		self.reset()

	def reset(self):
		self.altitude = 1500.0
		self.vitesse_verticale = 0.0
		return self._get_state()

	def _get_state(self):
		alt_norm = (self.altitude - self.altitude_cible) / 1000.0
		vit_norm = self.vitesse_verticale / 20.0
		return np.array([alt_norm, vit_norm], dtype=np.float32)

	def step(self, action):
		pousse = (action - 1) * 5.0
		gravite = -9.81 / 10.0

		self.vitesse_verticale += pousse + gravite
		self.vitesse_verticale *= 0.9
		self.altitude += self.vitesse_verticale
		ecart = abs(self.altitude_cible - self.altitude)
		recompense = -ecart / 100.0
		if ecart < 20.0:
			recompense += 5.0
		termine = self.altitude <= 0 or self.altitude> 10000.0
		if termine and self.altitude <= 0:
			recompense -= 100.0
		return self._get_state(), recompense, termine

class DQN(nn.Module):

	def __init__(self, input_dim=2, output_dim=3):
		super(DQN, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, output_dim)
		)

	def forward(self, x):
		return self.fc(x)

class DQNAgent:
	def __init__(self):
		self.model = DQN()
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()
		self.memory = deque(maxlen=10000)
		self.gamma = 0.99
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.batch_size = 64

	def choisir_action(self, state):
		if random.random() < self.epsilon:
			return random.randint(0, 2)
		state_t  = torch.FloatTensor(state).unsqueeze(0)
		with torch.no_grad():
			q_values = self.model(state_t)
		return torch.argmax(q_values).item()

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def train_step(self):
		if len(self.memory) < self.batch_size:
			return

		batch = random.sample(self.memory, self.batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)

		states_t = torch.FloatTensor(np.array(states))
		actions_t = torch.LongTensor(actions).unsqueeze(1)
		rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
		next_states_t = torch.FloatTensor(np.array(next_states))
		dones_t = torch.FloatTensor(dones).unsqueeze(1)

		current_q = self.model(states_t).gather(1, actions_t)
		max_next_q = self.model(next_states_t).max(1)[0].unsqueeze(1)
		target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

		loss = self.criterion(current_q, target_q.detach())
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

env = AvionEnv(altitude_cible=3000.0)
agent = DQNAgent()

print("----BEGIN OF the DQN TRAIN----")
for ep in range(200):
	state = env.reset()
	total_reward = 0
	for t in range(200):
		action = agent.choisir_action(state)
		next_state, reward, done = env.step(action)
		agent.remember(state, action, reward, next_state, done)
		agent.train_step()
		state = next_state
		total_reward += reward
		if done:
			break

	if (ep + 1) % 20 == 0:
		print(f"Episode {ep+1:03d}/200 | Score : {total_reward:.1f} | Final Alt : {env.altitude:.1f}m | Epsilon : {agent.epsilon:.2f}")
torch.save(agent.model.state_dict(), "modele_avion.pt")
print("Model saved to 'modele_avion.pt'")
print("\n--- Test flight with the trained agent ---")
agent.epsilon = 0.0
state = env.reset()
for t in range(150):
	action = agent.choisir_action(state)
	next_state, reward, done = env.step(action)
	state = next_state
	if t % 15 == 0 or t == 149:
		print(f"t={t:03d}s | Alt: {env.altitude:.1f}m | Speed: {env.vitesse_verticale:.1f}m/s")
	if done:
		break
