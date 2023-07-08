import joblib
import pandas as pd
import numpy as np


from sklearn.linear_model import LinearRegression


model1 = joblib.load('models/modelLR.joblib')
new_data = pd.DataFrame([[14.6,39.31,1011.11,72.52]], columns=['AT', 'V', 'AP', 'RH'])
prediction1 = model1.predict(new_data)
print(f"LR prediction {prediction1}")
model2 = joblib.load('models/modelRF.joblib')
prediction2 = model2.predict(new_data)
print(f"RF prediction {prediction2}")





