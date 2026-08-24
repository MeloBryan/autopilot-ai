import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque

class AvionEnv2D:

	def __init__(self, target_x=5000.0, target_y=3000.0):
		self.target_x = target_x
		self.target_y = target_y
		self.reset()

	def reset(self):
		self.x = 0.0
		self.y = 1000.0
		self.vx = 50.0
		self.vy = 0.0
		self.angle = 0.0
		return self._get_state()

	def _get_state(self):
		dx_norm = (self.target_x - self.x) / 5000.0
		dy_norm = (self.target_y -self.y) / 3000.0
		vx_norm = self.vx / 100.0
		vy_norm = self.vy / 50.0
		angle_norm = self.angle / math.pi
		return np.array([dx_norm, dy_norm, vx_norm, vy_norm, angle_norm], dtype=np.float32)

	def step(self, action):
		if action == 0:
			self.angle += 0.05
		if action == 2:
			self.angle -= 0.05

		self.angle = np.clip(self.angle, -math.pi/4, math.pi/4)

		poussee = 25.0
		gravite = 9.81
		frottement = 0.98

		portance = (self.vx * 0.2) * math.sin(self.angle)

		ax = poussee * math.cos(self.angle)
		ay = poussee * math.sin(self.angle) + portance - gravite

		self.vx = (self.vx + ax * 0.1) * frottement
		self.vy = (self.vy + ay * 0.1) * frottement

		self.x += self.vx * 0.1 * 10
		self.y += self.vy * 0.1 * 10

		dist_precedante = math.hypot(self.target_x - (self.x - self.vx), self.target_y - (self.y - self.vy))
		dist_actuelle = math.hypot(self.target_x - self.x, self.target_y- self.y)

		termine = self.y <= 0 or self.y > 10000.0 or self.x > 10000.0 or self.x < -1000.0
		recompense = (dist_precedante - dist_actuelle) / 5.0

		if self.y < self.target_y and self.vx > 0:
			recompense += 1.0

		if dist_actuelle < 150.0:
			recompense += 500.0
			termine = True
		elif termine:
			recompense -= 200.0

		return self._get_state(), recompense, termine

class DQN2D(nn.Module):

	def __init__(self, input_dim=5, output_dim=3):
		super(DQN2D, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(input_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, output_dim)
		)

	def forward(self, x):
		return self.fc(x)

class DQNAgent2D:

	def __init__(self):
		self.model = DQN2D()
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
		self.criterion = nn.MSELoss()
		self.memory = deque(maxlen=20000)
		self.gamma = 0.99
		self.epsilon = 1.0
		self.epsilon_min = 0.02
		self.epsilon_decay = 0.992
		self.batch_size = 64

	def choisir_action(self, state):
		if random.random() < self.epsilon:
			return random.randint(0, 2)
		state_t = torch.FloatTensor(state).unsqueeze(0)
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

env = AvionEnv2D(target_x=5000.0, target_y=3000.0)
agent = DQNAgent2D()

print("----BEGIN OF the DQN TRAIN----")
for ep in range(300):
	state = env.reset()
	total_reward = 0
	for t in range(300):
		action = agent.choisir_action(state)
		next_state, reward, done = env.step(action)
		agent.remember(state, action, reward, next_state, done)
		agent.train_step()
		state = next_state
		total_reward += reward
		if done:
			break

	if (ep + 1) % 30 == 0:
		dist = math.hypot(env.target_x - env.x, env.target_y - env.y)
		print(f"Episode {ep+1:03d}/300 | Score: {total_reward:.1f} | Final Pos: X={env.x:.0f}m, Y={env.y:.0f}m | Target Dist: {dist:.0f}m")

torch.save(agent.model.state_dict(), "model_avion_2d.pt")
print("2D model saved to 'modele_avion_2d.pt'")

print("\n--- 2D TEST FLIGHT ---")
agent.epsilon = 0.0
state = env.reset()
for t in range(200):
	action = agent.choisir_action(state)
	next_state, reward, done = env.step(action)
	state = next_state
	if t % 20 == 0 or done:
		dist = math.hypot(env.target_x - env.x, env.target_y - env.y)
		print(f"t={t:03d}s | Pos: ({env.x:.0f}m, {env.y:.0f}m) | Angle: {math.degrees(env.angle):.1f}° | Target Dist: {dist:.0f}m")
	if done:
		print("Hit or End of Flight!")
		break
