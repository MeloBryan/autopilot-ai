import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class Airplane3DEnv:
	"""
	3D Airplane Environment for Reinforcement Learning.
	"""
	def __init__(self, num_envs=16):
		self.num_envs = num_envs
		self.dt = 1.0
		self.velocity = 50.0
		self.waypoint_count = 3
		self.waypoint_x_range = (1500.0, 9000.0)
		self.waypoint_y_range = (-4000.0, 4000.0)
		self.waypoint_z_range = (500.0, 4500.0)
		self.waypoint_rng = np.random.default_rng()
		self._waypoints = np.zeros(
			(self.num_envs, self.waypoint_count, 3), dtype=np.float32
		)
		self.waypoint_radius = 150.0
		self.max_steps = 500

		self.x = np.zeros(num_envs, dtype=np.float32)
		self.y = np.zeros(num_envs, dtype=np.float32)
		self.z = np.zeros(num_envs, dtype=np.float32)
		self.pitch = np.zeros(num_envs, dtype=np.float32)
		self.yaw = np.zeros(num_envs, dtype=np.float32)
		self.roll = np.zeros(num_envs, dtype=np.float32)
		self.current_wp_idx = np.zeros(num_envs, dtype=np.int32)
		self.last_wp_idx = np.zeros(num_envs, dtype=np.int32)
		self.steps = np.zeros(num_envs, dtype=np.int32)
		self.previous_distances = np.zeros(num_envs, dtype=np.float32)

		self.reset_all()

	@property
	def waypoints(self):
		"""Waypoints of the first environment, used by the evaluation plot."""
		return [waypoint for waypoint in self._waypoints[0]]

	def _generate_waypoints(self, env_idx):
		"""Generate an ordered random route for one environment."""
		x_values = np.sort(self.waypoint_rng.uniform(
			*self.waypoint_x_range, self.waypoint_count
		))
		y_values = self.waypoint_rng.uniform(
			*self.waypoint_y_range, self.waypoint_count
		)
		z_values = self.waypoint_rng.uniform(
			*self.waypoint_z_range, self.waypoint_count
		)
		self._waypoints[env_idx] = np.column_stack(
			(x_values, y_values, z_values)
		).astype(np.float32)

	def reset_all(self):
		"""
		Reset all environments.
		"""
		for i in range(self.num_envs):
			self._generate_waypoints(i)

		self.x.fill(0.0)
		self.y.fill(0.0)
		self.z.fill(1000.0)
		self.pitch.fill(0.0)
		self.yaw.fill(0.0)
		self.roll.fill(0.0)
		self.current_wp_idx.fill(0)
		self.last_wp_idx.fill(0)
		self.steps.fill(0)
		self.previous_distances = np.linalg.norm(
			self._waypoints[:, 0] - np.column_stack((self.x, self.y, self.z)), axis=1
		).astype(np.float32)

		return self._get_states()

	def reset_single(self, i):
		"""
		Reset a single environment at index i.
		"""
		self._generate_waypoints(i)
		self.x[i] = 0.0
		self.y[i] = 0.0
		self.z[i] = 1000.0
		self.pitch[i] = 0.0
		self.yaw[i] = 0.0
		self.roll[i] = 0.0
		self.current_wp_idx[i] = 0
		self.steps[i] = 0
		self.previous_distances[i] = np.linalg.norm(
			self._waypoints[i, 0] - np.array([0.0, 0.0, 1000.0])
		)

	def _get_states(self):
		"""
		Get the current states for all environments.
		"""
		states = np.zeros((self.num_envs, 9), dtype=np.float32)
		for i in range(self.num_envs):
			target_idx = min(self.current_wp_idx[i], self.waypoint_count - 1)
			target = self._waypoints[i, target_idx]
			dx = target[0] - self.x[i]
			dy = target[1] - self.y[i]
			dz = target[2] - self.z[i]
			dist_horiz = np.sqrt(dx**2 + dy**2)
			dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)

			desired_yaw = np.arctan2(dy, dx)
			yaw_error = (desired_yaw - self.yaw[i] + np.pi) % (2 * np.pi) - np.pi

			desired_pitch = np.arctan2(dz, max(dist_horiz, 1.0))
			pitch_error = desired_pitch - self.pitch[i]

			states[i] = np.array([
				dx / 10000.0,
				dy / 10000.0,
				dz / 5000.0,
				dist_3d / 10000.0,
				self.pitch[i],
				self.yaw[i],
				self.roll[i],
				yaw_error,
				pitch_error
			], dtype=np.float32)

		return states
	
	def step(self, actions):
		"""
		Perform a step in the environment given an action.
		"""
		self.steps += 1
		rewards = np.zeros(self.num_envs, dtype=np.float32)
		dones = np.zeros(self.num_envs, dtype=bool)

		for i in range(self.num_envs):
			if actions[i] == 0:
				self.roll[i] = max(-np.radians(30), self.roll[i] - np.radians(5))
			elif actions[i] == 1:
				self.roll[i] *= 0.5
			elif actions[i] == 2:
				self.roll[i] = min(np.radians(30), self.roll[i] + np.radians(5))
			elif actions[i] == 3:
				self.pitch[i] = max(-np.radians(20), self.pitch[i] - np.radians(3))
			elif actions[i] == 4:
				self.pitch[i] = min(np.radians(20), self.pitch[i] + np.radians(3))

			g = 9.81
			yaw_rate = (g * np.tan(self.roll[i])) / self.velocity
			self.yaw[i] = (self.yaw[i] + yaw_rate * self.dt + np.pi) % (2 * np.pi) - np.pi
			self.x[i] += self.velocity * np.cos(self.pitch[i]) * np.cos(self.yaw[i]) * self.dt
			self.y[i] += self.velocity * np.cos(self.pitch[i]) * np.sin(self.yaw[i]) * self.dt
			self.z[i] += self.velocity * np.sin(self.pitch[i]) * self.dt
			self.z[i] = max(50.0, self.z[i])
			target_idx = min(self.current_wp_idx[i], self.waypoint_count - 1)
			target = self._waypoints[i, target_idx]
			dist_3d = np.linalg.norm(np.array([self.x[i], self.y[i], self.z[i]]) - target)
			reward = (self.previous_distances[i] - dist_3d) / 100.0
			self.previous_distances[i] = dist_3d

			state_curr = self._get_states()[i]
			yaw_err = abs(state_curr[7])
			pitch_err = abs(state_curr[8])
			alignement_bonus = max(0.0, 1.0 - (yaw_err + pitch_err)) * 2.0
			reward += alignement_bonus

			if dist_3d < self.waypoint_radius:
				precision_bonus = (1.0 - (dist_3d / self.waypoint_radius)) * 500.0
				reward += 500.0 + precision_bonus
				self.current_wp_idx[i] += 1
				if self.current_wp_idx[i] >= self.waypoint_count:
					reward += 1500.0
					dones[i] = True
				else:
					next_target = self._waypoints[i, self.current_wp_idx[i]]
					self.previous_distances[i] = np.linalg.norm(
						next_target - np.array([self.x[i], self.y[i], self.z[i]])
					)
			if self.steps[i] >= self.max_steps:
				dones[i] = True

			rewards[i] = reward

		for i in range(self.num_envs):
			self.last_wp_idx[i] = self.current_wp_idx[i]
			if dones[i]:
				self.reset_single(i)

		next_state = self._get_states()
	
		return next_state, rewards, dones

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

	def __init__(self, state_dim=9, action_dim=5, lr=1e-3, gamma=0.99, buffer_capacity=100000):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.gamma = gamma

		self.epsilon = 1.0
		self.epsilon_min = 0.05
		self.epsilon_decay = 0.9995

		self.memory = deque(maxlen=buffer_capacity)
		self.batch_size = 128

		if torch.cuda.is_available():
			self.device = torch.device("cuda")
		elif torch.backends.mps.is_available():
			self.device = torch.device("mps")
		else:
			self.device = torch.device("cpu")

		self.policy_net = DQN3D(state_dim, action_dim).to(self.device)
		self.target_net = DQN3D(state_dim, action_dim).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()

		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
		self.criterion = nn.SmoothL1Loss()

	def select_actions(self, states, evaluate=False):
		"""Action selection based on the epsilon-greedy policy"""
		if not evaluate and random.random() < self.epsilon:
			return np.random.randint(0, self.action_dim, size=len(states))

		state_t = torch.tensor(states, dtype=torch.float32, device=self.device)
		with torch.no_grad():
			q_values = self.policy_net(state_t)
		return torch.argmax(q_values, dim=1).cpu().numpy()

	def remember_batch(self, states, actions, rewards, next_states, dones):
		"""Adds a transition to the buffer"""
		for i in range(len(states)):
			self.memory.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))

	def train_step(self):
		"""Learning step using a sample from the Replay Buffer"""
		if len(self.memory) < self.batch_size:
			return None

		batch = random.sample(self.memory, self.batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)

		states_t = torch.tensor(np.array(states, dtype=np.float32), device=self.device)
		actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
		rewards_t = torch.tensor(rewards).unsqueeze(1).to(self.device)
		next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
		dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

		current_q = self.policy_net(states_t).gather(1, actions_t)

		with torch.no_grad():
			next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
			max_next_q = self.target_net(next_states_t).gather(1, next_actions)
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

# --- TRAINING ---
NUM_ENVS = 16
env = Airplane3DEnv(num_envs=NUM_ENVS)
agent = DQNAgent3D(state_dim=9, action_dim=5)

total_episodes_target = 10000
train_freq = 8
target_update_freq = 50
target_update_steps = 1000
step_counter = 0
completed_episodes = 0

states = env.reset_all()

print(f"Starting training for {total_episodes_target} episodes with {NUM_ENVS} parallel environments.")

while completed_episodes < total_episodes_target:
	actions = agent.select_actions(states)

	next_states, rewards, dones = env.step(actions)

	agent.remember_batch(states, actions, rewards, next_states, dones)

	step_counter += 1
	if step_counter % train_freq == 0:
		agent.train_step()
	if step_counter % target_update_steps == 0:
		agent.update_target_network()

	for i, done in enumerate(dones):
		if done:
			completed_episodes += 1
			if completed_episodes % target_update_freq == 0:
				print(f"Completed Episodes: {completed_episodes}/{total_episodes_target}, Epsilon: {agent.epsilon:.4f}")

	states = next_states

print("Training completed.")
torch.save(agent.policy_net.state_dict(), "airplane_dqn_3D.pt")
print("Trained model saved as 'airplane_dqn_3D.pt'.")

# --- EVALUATION ---

print("Starting evaluation of the trained agent.")
eval_env = Airplane3DEnv(num_envs=1)
evaluation_episodes = 20
max_eval_steps = 1000
successful_episodes = 0
validated_waypoints = []
trajectory_x = []
trajectory_y = []
trajectory_z = []

for episode in range(evaluation_episodes):
	state = eval_env.reset_all()
	episode_trajectory_x = [eval_env.x[0]]
	episode_trajectory_y = [eval_env.y[0]]
	episode_trajectory_z = [eval_env.z[0]]
	done = False
	eval_steps = 0

	while not done and eval_steps < max_eval_steps:
		eval_steps += 1
		action = agent.select_actions(state, evaluate=True)[0]
		next_state, reward, done_array = eval_env.step(np.array([action]))
		episode_trajectory_x.append(eval_env.x[0])
		episode_trajectory_y.append(eval_env.y[0])
		episode_trajectory_z.append(eval_env.z[0])
		state = next_state
		done = done_array[0]

	waypoints_reached = int(eval_env.last_wp_idx[0])
	validated_waypoints.append(waypoints_reached)
	if waypoints_reached == len(eval_env.waypoints):
		successful_episodes += 1
	if episode == 0:
		trajectory_x = episode_trajectory_x
		trajectory_y = episode_trajectory_y
		trajectory_z = episode_trajectory_z

print(
	 f"Evaluation completed: {successful_episodes}/{evaluation_episodes} successful episodes, "
	 f"average waypoints: {np.mean(validated_waypoints):.2f}/{len(eval_env.waypoints)}."
)

# --- GENERATE 3D TRAJECTORY PLOT ---

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_x, trajectory_y, trajectory_z, label='Airplane Trajectory', color='blue', linewidth=2)
wp_coords = np.array(eval_env.waypoints)
ax.scatter(wp_coords[:, 0], wp_coords[:, 1], wp_coords[:, 2], color='red', s=120, label='Waypoints 3D', zorder=5)
ax.scatter([0], [0], [1000], color='green', s=100, marker='^', label='Start Position', zorder=5)
for i, (wx, wy, wz) in enumerate(eval_env.waypoints):
	ax.text(wx, wy, wz + 150, f"WP{i+1}\n({int(wx)}, {int(wy)}, {int(wz)})", color='black', fontsize=9, fontweight='bold', ha='center')
ax.set_title('3D Airplane Trajectory', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('X Position', fontsize=12)
ax.set_ylabel('Y Position', fontsize=12)
ax.set_zlabel('Z Position', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('airplane_trajectory_3D.png', dpi=300)
plt.close()
print("3D trajectory plot saved as 'airplane_trajectory_3D.png'.")