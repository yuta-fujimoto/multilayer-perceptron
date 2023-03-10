import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib
import matplotlib.pyplot as  plt
import argparse
import json

import layer
from model import Sequence

def plot_history(history):
	x = np.arange(1, history['epoch'] + 1)
	fig, axes = plt.subplots(2, 2)
	axes[0][0].plot(x, history['train_loss'], label='train', alpha=0.7, linewidth=0.5)
	axes[0][0].plot(x, history['valid_loss'], label='validation', alpha=0.7, linewidth=0.5)
	axes[0][0].legend()
	axes[0][0].set_xlabel('epoch')
	axes[0][0].set_ylabel('loss')
	axes[0][0].grid(True)

	axes[0][1].plot(x, history['train_accuracy'], label='train', alpha=0.7, linewidth=0.5)
	axes[0][1].plot(x, history['valid_accuracy'], label='validation', alpha=0.7, linewidth=0.5)
	axes[0][1].legend()
	axes[0][1].set_xlabel('epoch')
	axes[0][1].set_ylabel('accuracy')
	axes[0][1].grid(True)

	axes[1][0].plot(x, history['train_false_negative'], label='train', alpha=0.7, linewidth=0.5)
	axes[1][0].plot(x, history['valid_false_negative'], label='validation', alpha=0.7, linewidth=0.5)
	axes[1][0].legend()
	axes[1][0].set_xlabel('epoch')
	axes[1][0].set_ylabel('false negative rate')
	axes[1][0].grid(True)

	axes[1][1].axis('off')

	plt.tight_layout()
	fig.savefig('history.png')

if __name__ == '__main__':
	np.random.seed(42)
	parser = argparse.ArgumentParser()
	parser.add_argument('--filepath', default='data.csv', help='data filepath(csv)')
	parser.add_argument('--out', dest='out', type=str, default=None, help='output json name')
	args = parser.parse_args()

	df = pd.read_csv(args.filepath)
	attirbutes = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
				'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

	columns = ['ID', 'Diagnosis']
	for s in attirbutes:
		columns.append(f'Mean {s}')
		columns.append(f'{s} SE')
		columns.append(f'Worst {s}')
	df.columns = columns

	X = df.drop(columns=['ID', 'Diagnosis']).values

	oheDiagnosis = OneHotEncoder()
	Y = oheDiagnosis.fit_transform(df[['Diagnosis']]).toarray()

	n_layers = X.shape[1]

	model = Sequence([
		layer.Dense(n_layers, 30, layer.Relu),
		layer.Dense(30, 30, layer.Relu),
		layer.Dense(30, 2, layer.Softmax),
	])

	model.compile(loss='binaryCrossEntropy')

	# debug
	history = model.fit(X, Y, epoch=1000, learning_rate=0.1, early_stopping=True)

	# save model and encoder
	joblib.dump(model, 'params/model.joblib')
	joblib.dump(oheDiagnosis, 'params/ohe.joblib')
	if args.out != None:
		with open(args.out, 'w') as f:
			json.dump(history, f, ensure_ascii=True)

	plot_history(history)
