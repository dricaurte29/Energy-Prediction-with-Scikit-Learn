import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('data/CCPP_data.csv')

# Split the dataset into features (X) and target variable (y)
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()

# Train the Linear Regression model
model.fit(X_train, y_train)

# Save the trained Linear Regression model to a file
joblib.dump(model, 'models/modelLR.joblib')

# Make predictions on the test set using the Linear Regression model
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) for Linear Regression
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"LR  MSE: {mse}, MAE: {mae}")

# Random Forest Model
model_rf = RandomForestRegressor()

# Train the Random Forest model
model_rf.fit(X_train, y_train)

# Save the trained Random Forest model to a file
joblib.dump(model_rf, 'models/modelRF.joblib')

# Make predictions on the test set using the Random Forest model
y_pred = model_rf.predict(X_test)

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) for Random Forest
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"RF  MSE: {mse}, MAE: {mae}")
