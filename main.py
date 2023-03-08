import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import Layer
from Model import Sequence

np.random.seed(42)
df = pd.read_csv('data.csv')

attirbutes = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
              'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

columns = ['ID', 'Diagnosis']
for s in attirbutes:
    columns.append(f'Mean {s}')
    columns.append(f'{s} SE')
    columns.append(f'Worst {s}')
df.columns = columns

features = ['Worst Area', 'Worst Smoothness', 'Mean Texture']
X = df[features].values
# X = df.drop(columns=['ID', 'Diagnosis']).values

oheDiagnosis = OneHotEncoder()
Y = oheDiagnosis.fit_transform(df[['Diagnosis']]).toarray()

n_layers = X.shape[1]

model = Sequence([
    Layer.Dense(n_layers, 4, Layer.Relu),
    # Layer.Dense(4, 4, Layer.Relu),
    # Layer.Dense(4, 4, Layer.Relu),
    Layer.Dense(4, 2, Layer.Softmax),
])

model.compile(loss='binaryCrossEntropy')

# debug
model.fit(X, Y, epoch=2)

grad = np.array(model.debugCache['backprop'], dtype=float)
gradapprox = np.array(model.debugCache['approx'])

numerator = np.linalg.norm(grad[0] - gradapprox[0])                                     # Step 1'
denominator = np.linalg.norm(grad[0]) + np.linalg.norm(gradapprox[0])                   # Step 2'
difference = numerator / denominator                                              # Step 3'

print('difference:', difference)

# sample_model = Sequence([
#     layer 
# ])
# sample_model.fit(sample_X, sample_Y, epoch=1)


# わからん～～～～～～～
