import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data from csv and prepare training set
df = pd.read_csv('datasets/position-salaries-dataset/Position_Salaries.csv')
print(df)
print('----------------------------------------------------')
X = df.iloc[:, 1:2]
y = df.iloc[:, -1]

# Data preprocessing step
poly_reg = PolynomialFeatures(degree=2)  # degree greater than 2 leads to overfitting
X_poly_train = poly_reg.fit_transform(X)
X_test = [[2.3], [3.6], [4.4], [5.5], [5.9], [7.34], [7.98], [8.23], [9.5], [9.885]]
X_poly_test = poly_reg.fit_transform(X_test)

# Fit model
linear_regressor = LinearRegression()
linear_regressor.fit(X_poly_train, y)

# Test and evaluate model
y_predicted = linear_regressor.predict(X_poly_test)
print('r^2 score is {}'.format(r2_score(y, y_predicted)))

# Plot training and testing sets
plt.scatter(X, y, color='red')
plt.plot(X, y, color='red')
plt.scatter(X_test, y_predicted, color='blue')
plt.plot(X_test, y_predicted, color='blue')
plt.show()
