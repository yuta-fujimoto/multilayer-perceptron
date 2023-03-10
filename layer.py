import numpy as np
import math

class Dense:
	def __init__(self, n_input, n_output, acitivation) -> None:
		self.acitivation = acitivation
		# he initialization
		# the larger n_input is, the samller weight should be
		self.W = np.random.randn(n_output, n_input) * math.sqrt(2./n_input)
		self.b = np.zeros((n_output, 1))
		self.A = None # (output_shape, n_samples)
		self.A_prev = None # (input_shape, n_samples)
		self.epsilon = 0.01

	def set_params(self, leaning_rate):
		self.leaning_rate = leaning_rate

	def forward(self, A_prev):
		Z = np.dot(self.W, A_prev) + self.b
		A = self.acitivation.forward(Z)

		self.A = A
		self.Z = Z
		self.A_prev = A_prev

		return A

	def backwardOutputLayer(self, Y, n_samples):
		dZ = -1. * (Y - self.A)

		dW = np.dot(dZ, self.A_prev.T) / n_samples
		db = np.sum(dZ, axis=1, keepdims=True) / n_samples
		W = np.copy(self.W)

		self.W = self.W - self.leaning_rate * dW
		self.b = self.b - self.leaning_rate * db

		return W, dZ

	def backwardHiddenLayer(self, W_next, dZ_next, n_samples):
		dZ = np.dot(W_next.T, dZ_next) * self.acitivation.backward(self.Z)

		dW = np.dot(dZ, self.A_prev.T) / n_samples
		db = np.sum(dZ, axis=1, keepdims=True) / n_samples
		W = np.copy(self.W)

		self.W = self.W - self.leaning_rate * dW
		self.b = self.b - self.leaning_rate * db

		return W, dZ

class Relu:
	def __init__(self) -> None:
		pass

	def forward(Z):
		return np.maximum(0., Z)

	def backward(Z):
		return 1.0 * (Z > 0.)

class Softmax:
	def __init__(self) -> None:
		pass

	def forward(Z):
		# avoid inf overflow
		expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
		return expZ / np.sum(expZ, axis=0, keepdims=True)

	def backward(Z):
		return Z * (1. - Z)
