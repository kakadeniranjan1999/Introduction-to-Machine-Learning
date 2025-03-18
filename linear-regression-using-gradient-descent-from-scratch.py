import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# Read csv and prepare training data
data = pd.read_csv('datasets/salary-data-dataset/Salary_Data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
print(data)

# Fit regression line using Gradient Descent
m = 0
c = 0
L = 0.0001  # INCREASING L AND KEEPING EPOCHS CONSTANT WILL INCREASE r^2
epochs = 1000  # INCREASING EPOCHS AND KEEPING L CONSTANT WILL INCREASE r^2
n = float(len(X))
for i in range(epochs):
    Y_pred = m * X + c
    D_m = (-2 / n) * sum(X * (Y - Y_pred))
    D_c = (-2 / n) * sum(Y - Y_pred)
    m = m - L * D_m
    c = c - L * D_c
print('Slope is {} and y-intercept is {}'.format(m, c))

# Test model
Y_pred = m * X + c

# Plot data and regression line
plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.savefig('salary-data.png')
plt.show()

# Evaluate the model
print('r^2 score is {}'.format(r2_score(Y_pred, Y)))
