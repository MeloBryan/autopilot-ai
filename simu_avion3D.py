import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Airplane3DEnv:

	def __init__(self):
		self.dt = 1.0
		self.velocity = 50.0

		self.waypoints = [
			np.array([2000.0, 1500.0, 2000.0]),
			np.array([5000.0, -1000.0, 3500.0]),
			np.array([8000.0, 3000.0, 1500.0])
		]
		self.waypoint_radius = 150.0
		self.reset()

	def reset(self):
		self.x = 0.0
		self.y = 0.0
		self.z = 1000.0

		self.pitch = 0.0
		self.yaw = 0.0
		self.roll = 0.0

		self.current_wp_idx = 0
		self.steps = 0
		self.max_steps = 500

		return self._get_state()

	def _get_state(self):
		target_idx = min(self.current_wp_idx, len(self.waypoints) - 1)
		target = self.waypoints[target_idx]
		dx = target[0] - self.x
		dy = target[1] - self.y
		dz = target[2] - self.z
		dist_horiz = np.sqrt(dx**2 + dy**2)
		dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)

		desired_yaw = np.arctan2(dy, dx)
		yaw_error = (desired_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi

		desired_pitch = np.arctan2(dz, max(dist_horiz, 1.0))
		pitch_error = desired_pitch - self.pitch

		return np.array([
			dx / 10000.0,
			dy / 10000.0,
			dz / 5000.0,
			dist_3d / 10000.0,
			self.pitch,
			self.yaw,
			self.roll,
			yaw_error,
			pitch_error
		], dtype=np.float32)

	def step(self, action):
		self.steps += 1

		if action == 0:
			self.roll = max(-np.radians(30), self.roll - np.radians(5))
		elif action == 1:
			self.roll *= 0.5
		elif action == 2:
			self.roll = min(np.radians(30), self.roll + np.radians(5))
		elif action == 3:
			self.pitch = max(-np.radians(20), self.pitch - np.radians(3))
		elif action == 4:
			self.pitch = min(np.radians(20), self.pitch + np.radians(3))

		g = 9.81
		yaw_rate = (g * np.tan(self.roll)) / self.velocity
		self.yaw = (self.yaw + yaw_rate * self.dt + np.pi) % (2 * np.pi) - np.pi
		self.x += self.velocity * np.cos(self.pitch) * np.cos(self.yaw) * self.dt
		self.y += self.velocity * np.cos(self.pitch) * np.sin(self.yaw) * self.dt
		self.z += self.velocity * np.sin(self.pitch) * self.dt
		self.z = max(50.0, self.z)
		target_idx = min(self.current_wp_idx, len(self.waypoints) - 1)
		target = self.waypoints[target_idx]
		dist_3d = np.linalg.norm(np.array([self.x, self.y, self.z]) - target)
		reward = - (dist_3d / 500.0)

		state_curr = self._get_state()
		yaw_err = abs(state_curr[7])
		pitch_err = abs(state_curr[8])
		alignement_bonus = max(0.0, 1.0 - (yaw_err + pitch_err)) * 2.0
		reward += alignement_bonus

		done = False
		wp_validated = False
		tight_radius = 150.0

		if dist_3d < self.waypoint_radius:
			precision_bonus = (1.0 - (dist_3d / tight_radius)) * 500.0
			reward += 500.0 + precision_bonus
			self.current_wp_idx += 1
			wp_validated = True
			if self.current_wp_idx >= len(self.waypoints):
				reward += 1000.0
				done = True
		if self.steps >= self.max_steps:
			done = True
		return self._get_state(), reward, done, self.current_wp_idx, wp_validated

class DQN3D(nn.Module):

	def __init__(self, state_dim=9, action_dim=5):
		super(DQN3D, self).__init__()
		self.fc1 = nn.Linear(state_dim, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 64)
		self.out = nn.Linear(64, action_dim)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.relu(self.fc3(x))
		return self.out(x)

class DQNAgent3D:

	def __init__(self, state_dim=9, action_dim=5, lr=1e-3, gamma=0.99, buffer_capacity=50000):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.gamma = gamma

		self.epsilon = 1.0
		self.epsilon_min = 0.05
		self.epsilon_decay = 0.9992

		self.memory = deque(maxlen=buffer_capacity)
		self.batch_size = 64

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.policy_net = DQN3D(state_dim, action_dim).to(self.device)
		self.target_net = DQN3D(state_dim, action_dim).to(self.device)
		self.target_net.load_state_state_dict = self.policy_net.state_dict()
		self.target_net.eval()

		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
		self.criterion = nn.SmoothL1Loss()

	def select_action(self, state, evaluate=False):
		"""Action selection based on the epsilon-greedy policy"""
		if not evaluate and random.random() < self.epsilon:
			return random.randint(0, self.action_dim - 1)

		state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		with torch.no_grad():
			q_values = self.policy_net(state_tensor)
		return torch.argmax(q_values).item()

	def remember(self, state, action, reward, next_state, done):
		"""Adds a transition to the buffer"""
		self.memory.append((state, action, reward, next_state, done))

	def train_step(self):
		"""Learning step using a sample from the Replay Buffer"""
		if len(self.memory) < self.batch_size:
			return None

		batch = random.sample(self.memory, self.batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)

		states_t = torch.FloatTensor(np.array(states)).to(self.device)
		actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
		rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
		next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
		dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

		current_q = self.policy_net(states_t).gather(1, actions_t)

		with torch.no_grad():
			max_next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
			target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

		loss = self.criterion(current_q, target_q)
		self.optimizer.zero_grad()
		loss.backward()

		torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
		self.optimizer.step()

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		return loss.item()

	def update_target_network(self):
		"""Periodic update of the Target Network"""
		self.target_net.load_state_dict(self.policy_net.state_dict())

	def save(self, filepath="modele_airplane_3d.pt"):
		torch.save(self.policy_net.state_dict(), filepath)

	def load(self, filepath="modele_airplane3d.pt"):
		self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
		self.update_target_network()

env = Airplane3DEnv()
agent = DQNAgent3D(state_dim=9, action_dim=5)

num_episodes = 1000
target_update_freq = 10

scores = []
wp_achieved_history = []

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    done = False
    step_counter = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, current_wp_idx, wp_validated = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        step_counter += 1

        if step_counter % 4 == 0:
            agent.train_step()

        state = next_state
        total_reward += reward

    if episode % target_update_freq == 0:
        agent.update_target_network()

    scores.append(total_reward)
    wp_achieved_history.append(env.current_wp_idx)

    if episode % 50 == 0 or episode == num_episodes:
        print(f"Épisode {episode}/{num_episodes} - Score: {total_reward:.1f} - Waypoints: {env.current_wp_idx}/3 - Epsilon: {agent.epsilon:.3f}")

# --- 5. ÉVALUATION ET PLOT 3D ---
state = env.reset()
trajectory_x = [env.x]
trajectory_y = [env.y]
trajectory_z = [env.z]
done = False

while not done:
    action = agent.select_action(state, evaluate=True)
    next_state, reward, done, current_wp_idx, wp_validated = env.step(action)
    trajectory_x.append(env.x)
    trajectory_y.append(env.y)
    trajectory_z.append(env.z)
    state = next_state

print(f"Évaluation finale : {env.current_wp_idx}/3 Waypoints validés en 3D !")

# Génération de la figure de trajectoire 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Tracer de la trajectoire
ax.plot(trajectory_x, trajectory_y, trajectory_z, label="Trajectoire IA 3D", color="#1f77b4", linewidth=2.5)

# Tracer des Waypoints en 3D avec sphères/points
wp_coords = np.array(env.waypoints)
ax.scatter(wp_coords[:, 0], wp_coords[:, 1], wp_coords[:, 2], color='red', s=120, label='Waypoints 3D', zorder=5)

# Départ
ax.scatter([0], [0], [1000], color='green', s=100, marker='^', label='Départ')

for i, (wx, wy, wz) in enumerate(env.waypoints):
    ax.text(wx, wy, wz + 150, f"WP{i+1}\n({int(wx)}, {int(wy)}, {int(wz)})", color='black', fontsize=9, fontweight='bold', ha='center')

ax.set_title("Navigation Autopilote 3D - Multi-Waypoints (DQN)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("X - Est (m)")
ax.set_ylabel("Y - Nord (m)")
ax.set_zlabel("Z - Altitude (m)")
ax.legend(loc="upper left")
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("trajectoire_vol_3d.png", dpi=300)
print("Graphique de trajectoire 3D sauvegardé sous 'trajectoire_vol_3d.png'.")
