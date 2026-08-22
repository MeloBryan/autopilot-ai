import numpy as np

class AvionEnv:
	def __init__(self, altitude_cible=3000.0):
		self.altitude_cible = altitude_cible
		self.reset()

	def reset(self):
		self.altitude = 1500.0
		self.vitesse_verticale = 0.0
		return np.array([self.altitude, self.vitesse_verticale])

	def step(self, action):
		"""
		action : 0 (Piquer/Descendre), 1 (Maintenir), 2 (Cabrer/Monter)
		"""
		pousse = (action - 1) * 5.0
		gravite = -9.81 / 10.0

		self.vitesse_verticale += pousse + gravite
		self.vitesse_verticale *= 0.9
		self.altitude += self.vitesse_verticale
		ecart = abs(self.altitude_cible - self.altitude)
		recompense = -ecart / 100.0
		termine = self.altitude <= 0 or self.altitude> 10000.0
		etat = np.array([self.altitude, self.vitesse_verticale])
		return etat, recompense, termine

def choisir_action(etat, altitude_cible):
	altitude, vitesse = etat[0], etat[1]
	ecart = altitude_cible - altitude
	vitesse_cible = np.clip(ecart / 15.0, -15.0, 15.0)
	if vitesse < vitesse_cible - 0.5:
		return 2
	elif vitesse > vitesse_cible + 0.5:
		return 0
	return 1

env = AvionEnv(altitude_cible=3000.0)
etat_actuel = env.reset()

print("----BEGIN OF THE FLIGHT----")
for seconde in range(150):
	action = choisir_action(etat_actuel, env.altitude_cible)
	nouvel_etat, recompense, termine = env.step(action)
	etat_actuel = nouvel_etat
	print(f"t={seconde:02d}s | Alt: {nouvel_etat[0]:.1f}m | Vitesse:{nouvel_etat[1]:.1f}m/s | Score: {recompense:.1f}")
	if termine:
		print("Crash !!")
		break
