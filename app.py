from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the models
rf_model = joblib.load('models/random_forest_model.joblib')
lr_model = joblib.load('models/linear_regression_model.joblib')
dt_model = joblib.load('models/decision_tree_model.joblib')
svr_model = joblib.load('models/svr_model.joblib')
xgb_model = joblib.load('models/xgboost_model.joblib')

# Load the saved model feature names
model_columns = joblib.load('models/model_columns.pkl')

# Load the unique values for dropdowns from the training data
data = pd.read_csv('dataset.csv')  # Update this path
area_choices = data['Area'].unique().tolist()
item_choices = data['Item'].unique().tolist()

@app.route('/')
def home():
    return render_template('index.html', area_choices=area_choices, item_choices=item_choices)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    area = request.form.get('area')
    item = request.form.get('item')
    year = int(request.form.get('year'))
    avg_rainfall = float(request.form.get('average_rainfall'))
    pesticides = float(request.form.get('pesticides'))
    avg_temp = float(request.form.get('avg_temp'))

    # Create DataFrame for the input
    input_data = pd.DataFrame({
        'Area': [area],
        'Item': [item],
        'Year': [year],
        'average_rain_fall_mm_per_year': [avg_rainfall],
        'pesticides_tonnes': [pesticides],
        'avg_temp': [avg_temp]
    })

    # One-hot encode the categorical variables
    input_data = pd.get_dummies(input_data, columns=['Area', 'Item'], drop_first=True)

    # Ensure the DataFrame has the same columns as the training data
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # If the column is missing, fill it with 0

    # Rearranging columns to match the model's training data
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Make predictions
    rf_prediction = rf_model.predict(input_data)[0]
    lr_prediction = lr_model.predict(input_data)[0]
    dt_prediction = dt_model.predict(input_data)[0]
    svr_prediction = svr_model.predict(input_data)[0]
    xgb_prediction = xgb_model.predict(input_data)[0]

    return render_template('index.html', 
                           area_choices=area_choices,
                           item_choices=item_choices,
                           rf_prediction=rf_prediction,
                           lr_prediction=lr_prediction,
                           dt_prediction=dt_prediction,
                           svr_prediction=svr_prediction,
                           xgb_prediction=xgb_prediction)

if __name__ == '__main__':
    app.run(debug=True)