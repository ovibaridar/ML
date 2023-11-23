import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
path = 'data sets/car driving risk analysis2.csv'
datas = pd.read_csv(path)

# Define features (X) and target (y) and specify feature names
x = datas[['speed']]
y = datas['risk']

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.40, random_state=1)

# Create and train the linear regression model
rg = LinearRegression()

rg.fit(xtrain, ytrain)

# Make predictions on a new data point (e.g., speed = 90)
prediction = rg.predict([[90]])
print("Prediction:", prediction[0])

# Calculate and print R-squared (R^2)
y_pred = rg.predict(xtest)
r2 = r2_score(ytest, y_pred)
print("R-squared:", r2)
print("M =", rg.coef_)
print("C =", rg.intercept_)

# Plot the regression line
plt.scatter(datas.speed, datas.risk, color='b', label='Data')
plt.plot(datas.speed, rg.predict(datas[['speed']]), color='r', label='Regression Line')
plt.legend()
plt.xlabel('Speed')
plt.ylabel('Risk')
plt.show()

