import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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

# features = ['Worst Area', 'Worst Smoothness', 'Mean Texture']
# X = df[features].values
X = df.drop(columns=['ID', 'Diagnosis']).values

oheDiagnosis = OneHotEncoder()
Y = oheDiagnosis.fit_transform(df[['Diagnosis']]).toarray()
X = (X - X.mean()) / X.std()


n_layers = X.shape[1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=30, activation='relu', input_dim=30),
    tf.keras.layers.Dense(units=30, activation='relu', input_dim=30),
    tf.keras.layers.Dense(units=2, activation='softmax'),
])

model.compile(loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(X, Y, epochs=100, validation_split=0.2)