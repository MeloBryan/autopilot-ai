import torch
from simu_avion import AvionEnv, DQN

env = AvionEnv(altitude_cible=3000.0)
model = DQN()

model.load_state_dict(torch.load("modele_avion.pt"))
model.eval()

state = env.reset()
print("--- FLIGHT WITH LOADED MODEL ---")

for t in range(150):
	state_t = torch.FloatTensor(state).unsqueeze(0)
	with torch.no_grad():
		q_values = model(state_t)
	action = torch.argmax(q_values).item()

	next_state, reward, done = env.step(action)
	state = next_state

	if t % 15 == 0 or t == 149:
		print(f"t={t:03d}s | Alt: {env.altitude:.1f}m | Speed: {env.vitesse_verticale:.1f}m/s")
	if done:
		break
