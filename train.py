import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Load your dataset
df = pd.read_csv('dataset.csv')  # Update with your actual dataset path

# Preprocessing
df = pd.get_dummies(df, columns=['Area', 'Item'], drop_first=True)  # One-hot encoding

# Feature and label split
X = df.drop(columns=['hg/ha_yield'])  # Features
y = df['hg/ha_yield']  # Label

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

svr_model = SVR()
svr_model.fit(X_train, y_train)

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

# Save the models
joblib.dump(rf_model, 'models/random_forest_model.joblib')
joblib.dump(lr_model, 'models/linear_regression_model.joblib')
joblib.dump(dt_model, 'models/decision_tree_model.joblib')
joblib.dump(svr_model, 'models/svr_model.joblib')
joblib.dump(xgb_model, 'models/xgboost_model.joblib')

# Save the model columns (feature names)
model_columns = X.columns.tolist()  # Get the feature names
joblib.dump(model_columns, 'models/model_columns.pkl')

print("Models and model columns saved successfully.")