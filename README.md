# Crop Yield Prediction Using Machine Learning

## Overview
This project is aimed at predicting crop yield based on various factors such as area, item (crop type), year, rainfall, pesticide usage, and average temperature. The prediction models are created using machine learning algorithms, including:
- Random Forest Regressor
- Linear Regression
- Decision Tree Regressor
- Support Vector Regressor (SVR)
- XGBoost Regressor

Additionally, a Flask web application is provided for making crop yield predictions through a user-friendly interface.

## Dataset
The dataset used in this project has the following columns:
- `Area`: The region where the crop is grown.
- `Item`: The crop type.
- `Year`: The year of data collection.
- `hg/ha_yield`: The yield in hectograms per hectare (target variable).
- `average_rain_fall_mm_per_year`: Average rainfall in millimeters per year.
- `pesticides_tonnes`: The amount of pesticide used in tonnes.
- `avg_temp`: The average temperature.

## Features
- **Multiple Models**: Several machine learning models are trained to predict the crop yield, and their performance is compared.
- **Flask Web App**: A simple web interface to input the features and receive yield predictions.
- **Model Comparison**: Compare the performance of different models using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.

## Installation

### Prerequisites
1. Python 3.x
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Requirements
You can install the required Python packages by running the following command:
```bash
pip install pandas scikit-learn xgboost joblib flask
#   C r o p - Y i e l d - P r e d i c t i o n - U s i n g - M a c h i n e - L e a r n i n g 
 
 #   C r o p - Y i e l d - P r e d i c t i o n - U s i n g - M a c h i n e - L e a r n i n g 
 
 #   C r o p - Y i e l d - P r e d i c t i o n - U s i n g - M L 
 
