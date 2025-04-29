import numpy as np
import pandas as pd
train = pd.read_csv('train.tsv', sep='\t', header=None)

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

X_test = pd.read_csv('test.tsv', sep='\t', header=None).values
y_pred = X_test @ w
pd.DataFrame(y_pred).to_csv('answer.tsv', sep='\t', header=False, index=False, float_format='%.8f')
