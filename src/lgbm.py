import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LGBpipeline:

    def __init__(self):
        self._model = None

    def train(self, X_train, y_train, params: dict, n_boost: int = 200):
        
        train_data = lgb.Dataset(X_train, label=y_train)
        self._model = lgb.train(params, train_data, n_boost)

    def grid_search_cv(self, X_train, y_train, param_grid: dict = None):

        if param_grid is None:
            param_grid = {
                'linear_tree': [True, False],
                'num_leaves': [10, 20, 30],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.8, 0.9, 1.0],
                'min_data_in_leaf': [5, 10, 20],
            }

        lgb_model = lgb.LGBMRegressor(objective='regression', metric='rmse', verbosity=-1)

        grid_search = GridSearchCV(
            estimator=lgb_model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=5,
            verbose=False,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        print(f"Best Parameters: {self.best_params}")
        print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

        self._model = lgb.LGBMRegressor(objective='regression', metric='rmse', **self.best_params)
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        assert self._model is not None, 'Model not defined'

        y_pred = self._model.predict(X_test)

        return y_pred

    def plot(self, x_train, y_train, x_test, y_true, y_pred, title: str = None, label_x: str = None, label_y: str = None):
        
        plt.figure(figsize=(14, 7))
        plt.plot(x_train, y_train, label='training data')
        plt.plot(x_test, y_true, label='actual values', color='green')
        plt.plot(x_test, y_pred, label='predicted values', linestyle='--', color='red')
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.legend()
        plt.show()

if __name__=='__main__':
    
    params = {'feature_fraction': 0.8, 'learning_rate': 0.05, 'linear_tree': True, 'min_data_in_leaf': 20, 'num_leaves': 10, 'verbosity':-1}

    df = pd.read_csv('datasets/air_passengers.csv')
    df['Month'] = df['Month'].apply(lambda x: int(x.split('-')[1]))
    df[f'lag_{12}'] = df['#Passengers'].shift(12)
    df = df.dropna()

    train = df[:-12]
    test = df[-12:]
    features = ['Month', 'lag_12']
    X_train, y_train = train[features], train['#Passengers']
    X_test, y_test = test[features], test['#Passengers']

    pipeline = LGBpipeline()
    pipeline.train(X_train, y_train, params)
    y_pred = pipeline.predict(X_test)

    print(r2_score(y_test, y_pred))

    pipeline.plot(X_train.index, y_train, X_test.index, y_test, y_pred, 'Passengers time series', label_y='#Passengers')