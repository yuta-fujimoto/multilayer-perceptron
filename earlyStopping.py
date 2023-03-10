import numpy as np
import copy

class EarlyStopping:
	def __init__(self, on, patience = 5) -> None:
		self.on = on
		self.min_val_loss = float('inf')
		self.layers = None

		self.patience = patience
		self.count = 0

	def check(self, val_loss = None, layers = None):
		if self.on == False:
			return False
		if self.min_val_loss > val_loss:
			self.min_val_loss = val_loss
			self.layers = copy.deepcopy(layers)
			self.count = 0
		else:
			self.count += 1
			if self.count >= self.patience:
				return True

		return False

	def load_layers(self):
		return self.layers
