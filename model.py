from sklearn.model_selection import train_test_split
import numpy as np
from earlyStopping import EarlyStopping

class Sequence:
	def __init__(self, layers) -> None:
		self.layers = layers
		self.loss = ''

	def __forward(self, X):
		input = X
		for l in self.layers:
			output = l.forward(input)
			input = output

		return output

	def __backward(self, Y, n_sampels):
		l = len(self.layers)
		W, dZ = self.layers[l - 1].backwardOutputLayer(Y, n_sampels)
		for i in reversed(range(l - 1)):
			W, dZ = self.layers[i].backwardHiddenLayer(W, dZ, n_sampels)

	def __loss(self, output, Y, n_samples):
		if self.loss == 'binaryCrossEntropy':
			# avoid log(0)
			return -1. * ((Y * np.log(output + 1e-8)).sum() / n_samples)
		else:
			print('Model.Sequence: unknown loss')

	def __accuracy(self, output, Y):
		return (output.argmax(0) == Y.argmax(0)).mean()

	def __false_neagative_rate(self, output, Y):
		return ((Y.argmax(0) == 1) * (output.argmax(0) == 0)).sum() / (Y.argmax(0) == 1).sum()

	def compile(self, loss):
		self.loss = loss

	def fit(self, X, Y, learning_rate = 0.01, epoch = 10, early_stopping = None):
		# X:  (n_samples, n_features)
		# Y:  (n_samples, 1)

		checker = EarlyStopping(early_stopping)

		for l in self.layers:
			l.set_params(learning_rate)

		# standardlization
		self.mean = np.mean(X, axis=0)
		self.std = np.std(X, axis=0)
		X_norm = (X - self.mean) / self.std

		train_losses = []
		train_accuracies = []
		train_false_negatives = []
		valid_losses = []
		valid_accuracies = []
		valid_false_negatives = []
		for i in range(epoch):
			train_x, valid_x, train_y, valid_y = train_test_split(X_norm, Y, train_size=0.8, shuffle=True)
			n_trains = train_y.shape[0]
			n_valids = valid_y.shape[0]

			# for ease of calculation
			train_x, train_y = train_x.T, train_y.T
			valid_x, valid_y = valid_x.T, valid_y.T

			self.__forward(train_x)
			self.__backward(train_y, n_trains)

			# training loss and accuracy
			train_output = self.__forward(train_x)
			train_loss = self.__loss(train_output, train_y, n_trains)
			train_accuracy = self.__accuracy(train_output, train_y)
			train_false_negative = self.__false_neagative_rate(train_output, train_y)

			# validation loss and accuracy
			valid_output = self.__forward(valid_x)
			valid_loss = self.__loss(valid_output, valid_y, n_valids)
			valid_accuracy = self.__accuracy(valid_output, valid_y)
			valid_false_negative = self.__false_neagative_rate(valid_output, valid_y)

			print(
				f'epoch {i + 1}/{epoch}: - loss: {train_loss:.4} - acc: {train_accuracy:.4} - val_loss: {valid_loss:.4} - val_acc: {valid_accuracy:.4}')

			if checker.check(valid_loss, self.layers):
				self.layers = checker.load_layers()
				print('early stopping was applied')
				epoch = i
				break

			train_losses.append(train_loss)
			train_accuracies.append(train_accuracy)
			train_false_negatives.append(train_false_negative)

			valid_losses.append(valid_loss)
			valid_accuracies.append(valid_accuracy)
			valid_false_negatives.append(valid_false_negative)

		return {
			'train_loss': train_losses,
			'train_accuracy': train_accuracies,
			'train_false_negative': train_false_negatives,
			'valid_loss': valid_losses,
			'valid_accuracy': valid_accuracies,
			'valid_false_negative': valid_false_negatives,
			'epoch': epoch,
		}

	def evaluate(self, X, Y):
		# standardlization
		X_norm = (X - self.mean) / self.std

		# for ease of calculation
		X_norm, Y = X_norm.T, Y.T

		output = self.__forward(X_norm)
		loss = self.__loss(output, Y, Y.shape[1])

		return (loss)
