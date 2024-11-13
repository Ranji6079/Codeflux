import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



data = pd.read_csv('datasets/data.csv')



X = data[['X']]

Y = data['Y']



model = LinearRegression()



model.fit(X, Y)



Y_pred = model.predict(X)



plt.scatter(X, Y, color='blue', label='Actual data')

plt.plot(X, Y_pred, color='red', label='Linear regression line')

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Linear Regression')

plt.legend()

plt.show()



print(f"Coefficient: {model.coef_[0]}")

print(f"Intercept: {model.intercept_}")

