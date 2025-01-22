import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

#https://medium.com/@simon.peter.mueller/overcoming-the-limitations-of-tree-based-models-in-time-series-forecasting-c2c5bd71a8f1

# Load the dataset
#df = pd.read_csv('datasets/air_passengers.csv', parse_dates=['Month'], index_col='Month')
df = pd.read_csv('datasets/air_passengers.csv')

df['Month'] = df['Month'].apply(lambda x: int(x.split('-')[1]))
df[f'lag_{12}'] = df['#Passengers'].shift(12)
df = df.dropna()

train = df[:-12]
test = df[-12:]
features = ['Month', 'lag_12']
X_train, y_train = train[features], train['#Passengers']
X_test, y_test = test[features], test['#Passengers']

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'linear_tree': True,  # Enable linear tree
    'num_leaves': 10,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'min_data_in_leaf': 5
}
train_data = lgb.Dataset(X_train, label=y_train)
gbm = lgb.train(params, train_data, num_boost_round=200)

y_pred = gbm.predict(X_test)
print(f'r2: {r2_score(y_test, y_pred)}')

# Plotting the actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(train.index, y_train, label='Training Data')
plt.plot(test.index, y_test, label='Actual Values', color='green')
plt.plot(test.index, y_pred, label='Predicted Values', linestyle='--', color='red')
plt.title('Air Passengers Forecast with Standard LightGBM and Month Feature')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

print(df.iloc[-16:])