import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
from collections import deque

class AvionEnv2D:

    def __init__(self, waypoints=None):
        if waypoints is None:
            self.waypoints = [(2000.0, 1500.0), (5000.0, 3000.0), (8000.0, 1200.0)]
        else:
            self.waypoints = waypoints
        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 1000.0
        self.vx = 50.0
        self.vy = 0.0
        self.angle = 0.0
        self.current_wp_idx = 0
        self.target_x, self.target_y = self.waypoints[self.current_wp_idx]
        return self._get_state()

    def _get_state(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)
        angle_cible = math.atan2(dy, dx)

        erreur_angle = math.atan2(math.sin(angle_cible - self.angle), math.cos(angle_cible - self.angle))

        dist_norm = min(dist / 10000.0, 1.0)
        err_angle_norm = erreur_angle / math.pi
        vx_norm = self.vx / 100.0
        vy_norm = self.vy / 50.0
        angle_norm = self.angle / (math.pi / 4.0)
        return np.array([dist_norm, err_angle_norm, vx_norm, vy_norm, angle_norm], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.angle += 0.08
        elif action == 2:
            self.angle -= 0.08

        self.angle = np.clip(self.angle, -math.pi/3, math.pi/3)

        poussee = 25.0
        gravite = 9.81
        frottement = 0.98

        portance = (self.vx * 0.2) * math.sin(self.angle)
        ax = poussee * math.cos(self.angle)
        ay = poussee * math.sin(self.angle) + portance - gravite

        self.vx = (self.vx + ax * 0.1) * frottement
        self.vy = (self.vy + ay * 0.1) * frottement

        self.x += self.vx * 0.5
        self.y += self.vy * 0.5

        dist_actuelle = math.hypot(self.target_x - self.x, self.target_y - self.y)

        dx = self.target_x - self.x
        dy = self.target_y - self.y
        angle_cible = math.atan2(dy, dx)
        erreur_angle = abs(math.atan2(math.sin(angle_cible - self.angle), math.cos(angle_cible - self.angle)))

        recompense = 3.0 - (3.0 * (erreur_angle / math.pi)) - (dist_actuelle / 2000.0)

        termine = self.y <= 0 or self.y > 6000.0 or self.x > 12000.0 or self.x < -500.0

        if dist_actuelle < 250.0:
            recompense += 1000.0
            self.current_wp_idx += 1
            if self.current_wp_idx < len(self.waypoints):
                self.target_x, self.target_y = self.waypoints[self.current_wp_idx]
            else:
                recompense += 2000.0
                termine = True
        elif termine:
            recompense -= 1000.0

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
        self.target_model = DQN2D()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.993
        self.batch_size = 64
        self.train_step_count = 0

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

        with torch.no_grad():
            max_next_q = self.target_model(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % 500 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = AvionEnv2D()
agent = DQNAgent2D()

print("----BEGIN OF MULTI-WAYPOINT TRAIN----")
for ep in range(600):
    state = env.reset()
    total_reward = 0
    for t in range(500):
        action = agent.choisir_action(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state
        total_reward += reward
        if done:
            break

    agent.update_epsilon()

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1:03d}/600 | Score: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f} | WP Reach: {env.current_wp_idx}/{len(env.waypoints)} | Final Pos: X={env.x:.0f}m, Y={env.y:.0f}m")

torch.save(agent.model.state_dict(), "modele_avion_2d.pt")
print("Model saved to 'modele_avion_2d.pt'")

print("\n--- TEST FLIGHT MULTI-WAYPOINTS ---")
agent.epsilon = 0.0
state = env.reset()

history_x = [env.x]
history_y = [env.y]

for t in range(500):
    action = agent.choisir_action(state)
    next_state, reward, done = env.step(action)
    state = next_state

    history_x.append(env.x)
    history_y.append(env.y)

    wp_idx_safe = min(env.current_wp_idx, len(env.waypoints) - 1)
    if t % 25 == 0 or done:
        print(f"t={t:03d}s | Pos: ({env.x:.0f}m, {env.y:.0f}m) | Target WP{wp_idx_safe+1}: ({env.target_x:.0f}m, {env.target_y:.0f}m)")
    if done:
        print(f"End of Flight! Waypoints validated: {env.current_wp_idx}/{len(env.waypoints)}")
        break

plt.figure(figsize=(10, 5))
plt.plot(history_x, history_y, label="Aircraft trajectory", color='blue', linewidth=2)

wp_x = [wp[0] for wp in env.waypoints]
wp_y = [wp[1] for wp in env.waypoints]

plt.scatter(wp_x, wp_y, color='red', s=100, label="Waypoints", zorder=5)

for i, (wx, wy) in enumerate(env.waypoints):
    plt.annotate(f"WP{i+1}", (wx + 100, wy + 100), fontsize=10, fontweight='bold', color='darkred')

plt.axhline(0, color='black', linestyle='--', label="Ground")
plt.title("Autopilot AI - Multi-Waypoint Flight Path")
plt.xlabel("Horizontal distance X (m)")
plt.ylabel("Altitude Y (m)")
plt.grid(True)
plt.legend()
plt.savefig("trajectoire_vol.png")
print("Graph saved as 'trajectoire_vol.png'")
