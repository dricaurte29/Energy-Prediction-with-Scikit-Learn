# Energy Output Prediction with Regression Models

This project aims to predict the electrical energy output of a Combined Cycle Power Plant (CCPP) using regression models. The dataset used in this project contains features such as temperature, exhaust vacuum, ambient pressure, and relative humidity, which are used to predict the net hourly electrical energy output.

## Dataset
The dataset used for training and evaluation is stored in the `data/` directory. The `CCPP_data.csv` file contains the necessary input features and the corresponding target variable.

## Models
Trained regression models are saved in the `models/` directory. Two models are included:
- `modelLR.joblib`: Linear Regression model
- `modelRF.joblib`: Random Forest model

## Notebooks
The `notebooks/` directory contains Jupyter notebooks with the following functionalities:
- `Model_Training.ipynb`: Notebook for training the regression models using the dataset and saving the trained models.
- `Model_Prediction.ipynb`: Notebook for loading the trained models and making predictions on new data.

## Instructions
1. Install the required dependencies: pandas, numpy, scikit-learn.
2. Run the `Model_Training.ipynb` notebook to train the regression models and save them in the `models/` directory.
3. Use the `Model_Prediction.ipynb` notebook to load the trained models and make predictions on new data.

Feel free to explore the notebooks, modify the models, or experiment with different algorithms to enhance the prediction performance.

For further details, refer to the individual notebook files.

