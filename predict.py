import joblib
import pandas as pd

if __name__ == '__main__':
	model = joblib.load('model.joblib')
	oheDiagnosis = joblib.load('ohe.joblib')

	df = pd.read_csv('data.csv')

	attirbutes = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
				'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

	columns = ['ID', 'Diagnosis']
	for s in attirbutes:
		columns.append(f'Mean {s}')
		columns.append(f'{s} SE')
		columns.append(f'Worst {s}')
	df.columns = columns

	# features = ['Worst Area', 'Worst Smoothness', 'Mean Texture']
	# X = df[features].values
	X = df.drop(columns=['ID', 'Diagnosis']).values

	Y = oheDiagnosis.transform(df[['Diagnosis']]).toarray()

	loss = model.evaluate(X, Y)

	print('loss:', loss)
