import joblib
import pandas as pd
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--filepath', dest='filepath',  default='data.csv', help='data filepath(csv)')
	args = parser.parse_args()

	model = joblib.load('params/model.joblib')
	oheDiagnosis = joblib.load('params/ohe.joblib')

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
	Y = oheDiagnosis.transform(df[['Diagnosis']]).toarray()

	loss = model.evaluate(X, Y)
	print('loss:', loss)
