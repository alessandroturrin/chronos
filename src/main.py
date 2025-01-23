import pandas as pd  # requires: pip install pandas
import torch
from chronos.base import BaseChronosPipeline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
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

class MetaLearner:

    def __init__(self, chronos, lgb):
        self._chronos = chronos
        self._lgb = lgb

    def train(self, y, X_train, y_train):

        lgb_preds = self._lgb.predict(X_train)

        chronos_preds, _ = self._chronos.predict_quantiles(
                context=torch.tensor(y.values).clone().detach(),
                prediction_length=len(y_train),
                quantile_levels=[0.1, 0.5, 0.9],
            )
        chronos_preds = chronos_preds[0, :, 1]

        meta_features_train = np.column_stack([lgb_preds, chronos_preds])

        self._meta_model = LinearRegression()
        self._meta_model.fit(meta_features_train, y_train)
   
    def predict(self, y, X_test, y_test):

        lgb_preds = self._lgb.predict(X_test)

        chronos_preds, _ = self._chronos.predict_quantiles(
                context=torch.tensor(y.values).clone().detach(),
                prediction_length=len(y_test),
                quantile_levels=[0.1, 0.5, 0.9],
            )
        chronos_preds = chronos_preds[0, :, 1]

        meta_features_preds = np.column_stack([lgb_preds, chronos_preds])

        y_pred = self._meta_model.predict(meta_features_preds)

        return y_pred
    
chr_pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cpu",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)
lgb_pipeline = LGBpipeline()
params = {'feature_fraction': 0.8, 'learning_rate': 0.05, 'linear_tree': True, 'min_data_in_leaf': 20, 'num_leaves': 10, 'verbosity':-1}

df = pd.read_csv('datasets/air_passengers.csv')

df['Month'] = df['Month'].apply(lambda x: int(x.split('-')[1]))
df[f'lag_{12}'] = df['#Passengers'].shift(12)
df = df.dropna()

features = ['Month', 'lag_12']

train = int(len(df)*.5)
val = int(len(df)*.8)


X_train, y_train = df[features].iloc[:train], df['#Passengers'].iloc[:train]
X_val, y_val = df[features].iloc[train:val], df['#Passengers'].iloc[train:val]
X_test, y_test = df[features].iloc[val:], df['#Passengers'].iloc[val:]

lgb_pipeline = LGBpipeline()
lgb_pipeline.train(X_train, y_train, params)

ml = MetaLearner(chr_pipeline, lgb_pipeline)
ml.train(y_train, X_val, y_val)

y_pred = ml.predict(y_val, X_test, y_test)

r2 = r2_score(y_test, y_pred)

chronos_preds, _ = chr_pipeline.predict_quantiles(
                context=torch.tensor(df['#Passengers'].iloc[:val].values).clone().detach(),
                prediction_length=len(df['#Passengers'].iloc[val:]),
                quantile_levels=[0.1, 0.5, 0.9],
            )
chronos_preds = chronos_preds[0, :, 1]

r2_chr = r2_score(y_test, chronos_preds)

print(f'stack: {100*r2:.2f}%\nchronos: {100*r2_chr:.2f}%')

plt.figure(figsize=(14, 7))
plt.plot(np.arange(val), df['#Passengers'].iloc[:val].values, label='Training Data')
plt.plot(np.arange(start=val, stop=len(df)), df['#Passengers'].iloc[val:].values, label='Actual Values', color='green')
plt.plot(np.arange(start=val, stop=len(df)), chronos_preds, label='Chronos', linestyle='--', color='red')
plt.plot(np.arange(start=val, stop=len(df)), y_pred, label='Stack', linestyle='-', color='coral')
plt.title('Stack vs. Chronos')
plt.xlabel('time')
plt.ylabel('#Passengers')
plt.legend()
plt.show()