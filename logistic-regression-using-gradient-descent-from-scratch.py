import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score


def sigmoid(y_pred):
    y_predicted_new = []
    y_pred = -1 * y_pred
    for i in range(0, len(y_pred)):
        sig = 1 / (1 + math.exp(y_pred[i]))
        y_predicted_new.append(sig)
    plt.scatter(X, y_predicted_new)
    plt.plot(X, y_predicted_new)
    threshold = 0.6122  # keep in between 0.6122 and 0.6147    (max(y_predicted_new)+min(y_predicted_new))/2
    print('threshold=', threshold)
    y_predicted_binarized = []
    for i in y_predicted_new:
        if i < threshold:
            i = 0
            y_predicted_binarized.append(i)
        elif i >= threshold:
            i = 1
            y_predicted_binarized.append(i)
    return y_predicted_binarized


# Read csv and prepare training data
data = pd.read_csv('datasets/social-network-ads-dataset/Social_Network_Ads.csv')
X = data.iloc[:, 2].values
Y = data.iloc[:, 4].values

# Fit logistic regression model using Gradient Descent
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
plt.show()

# Apply Sigmoid to predicted values
y_predicted_binarized = sigmoid(Y_pred)

# Evaluate the model
matrix = confusion_matrix(Y, y_predicted_binarized)
print('Confusion Matrix=')
print(matrix)
print('r^2 score is {}'.format(r2_score(Y, y_predicted_binarized)))
