"""
Main module for the stack model composed by Chronos and LightGBM

Classes:
    LGBPipeline: to manage LightGBM regressor
    ChronosStack: to build Chronos stack model
"""

from argparse import ArgumentParser 

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb
from chronos.base import BaseChronosPipeline

from ext.preprocess_data import prepare_data
from ext.meta_datasets import *


def plot_results(df, stack_preds, chronos_preds, target: str, total_length: int, prediction_length: int):
    
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(start=0, stop=total_length-prediction_length), df[target].iloc[:total_length-prediction_length].values, label='data')
    plt.plot(np.arange(start=total_length-prediction_length, stop=total_length), df[target].iloc[total_length-prediction_length:].values, label='true values', color='green')
    plt.plot(np.arange(start=total_length-len(chronos_preds), stop=total_length), chronos_preds, label='Chronos', linestyle='--', color='red')
    plt.plot(np.arange(start=total_length-len(stack_preds), stop=total_length), stack_preds, label='ChronosStack', linestyle='-', color='coral')
    plt.title('Models predictions')
    plt.xlabel('time')
    plt.ylabel(target)
    plt.grid(visible=True)
    plt.legend()
    plt.show()

"""
Pipeline to manage training/grid search and predictions for LGB regressor
"""
class LGBpipeline:

    def __init__(self):
        self._model = None

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

    def train(self, X_train, y_train, params: dict = None, n_boost: int = 200):
        
        if params is None:
            params = {'feature_fraction': 0.8, 'learning_rate': 0.05, 'linear_tree': True, 
                        'min_data_in_leaf': 20, 'num_leaves': 31, 'verbosity':-1}

        train_data = lgb.Dataset(X_train, label=y_train)
        self._model = lgb.train(params, train_data, n_boost)

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


"""
Stack model for chronos and LGB regressor
"""
class ChronosStack:

    def __init__(self, chronos: str = 'amazon/chronos-bolt-small', lgb: LGBpipeline = None):
        
        self._chronos = BaseChronosPipeline.from_pretrained(
            chronos,
            device_map="cpu", 
            torch_dtype=torch.bfloat16,
        )

        if lgb is None:
            self._lgb = LGBpipeline()
        else:
            self._lgb = lgb
                
        self._scaler = MinMaxScaler()
        self._trained = False
    
    def get_chronos_prediction(
            self,
            context: torch.Tensor,
            prediction_length: int,
    ) -> np.array:

        self._chronos_predictions = {}
        
        chronos_preds, _ = self._chronos.predict_quantiles(
                context=context,
                prediction_length=prediction_length,
                quantile_levels=[0.1, 0.5, 0.9],
            )

        chronos_predictions = chronos_preds[0, :, 1].numpy()

        return chronos_predictions

    def fit(
            self, 
            data: dict, 
            grid_search: bool = False
        ) -> None:
        
        y_train_scaled = self._scaler.fit_transform(data['y_train'].reshape((-1,1))).flatten()
        
        if not grid_search:
            self._lgb.train(data['X_train'], y_train_scaled)
        else:
            self._lgb.grid_search_cv(data['X_train'], y_train_scaled)
        
        self._trained = True

    def predict(
            self,
            data: dict,
            prediction_length: int
        ) -> np.array:

        assert self._trained, 'Regressor not trained'

        lgb_predictions =  self._lgb.predict(data['X_test'])
        lgb_predictions = self._scaler.inverse_transform(lgb_predictions.reshape((1,-1)))[0]

        self._lgb_predictions = lgb_predictions

        residuals = data['y_test'] - lgb_predictions

        context = residuals[:-prediction_length]
        chronos_predictions = self.get_chronos_prediction(torch.tensor(context), prediction_length)

        predictions = lgb_predictions[-prediction_length:] + chronos_predictions

        return predictions
    

"""
Main
"""
if __name__=='__main__':

    parser = ArgumentParser()
    
    # let following params to use default settings
    parser.add_argument(
        '--dataset',
        help = 'select dataset(s) from available ones',
        required = False,
        default = ALL_DATASETS,
        choices = ALL_DATASETS,
        type = str,
        nargs='+'
    )

    parser.add_argument(
        '--model',
        help = 'select pre-trained chronos model',
        required = False,
        default = 'amazon/chronos-bolt-small',
        choices = ['amazon/chronos-bolt-tiny', 'amazon/chronos-bolt-small', 'amazon/chronos-bolt-base', 'amazon/chronos-bolt-large'],
        type = str
    )

    parser.add_argument(
        '--grid_search',
        help = 'run grid search for regressor',
        required = False,
        default = False,
        type = bool
    )

    parser.add_argument(
        '--prediction_length',
        help = 'length of chronos prediction (<=64)',
        required = False,
        default = None,
        type = int
    )

    parser.add_argument(
        '--train_size',
        help = 'size to split dataset (0<train<1)',
        required = False,
        default = None,
        type = float
    )


    # parse arguments
    args = parser.parse_args()

    datasets = args.dataset

    # init results
    results = {
        'dataset': list(),
        'model': list(),
        'r2': list(),
        'MSE': list()
    }

  
    # run
    for dataset in datasets:

        model = args.model
        grid_search = args.grid_search


        # load default values
        df, prediction_length, target, covariates, train_size = get_dataset(dataset)

        # check for edits in default data
        if args.prediction_length is not None:
            prediction_length = args.prediction_length

        if args.train_size is not None:
            train_size = args.train_size


        # create splits for train/test considering target and covariates
        data = prepare_data(df, target, covariates, train_size)

        # initialize chronos stack model
        stack = ChronosStack(model)


        # train stack learner (only predictor part)
        stack.fit(data, grid_search=grid_search)

        # get predictions and scores for stack
        y_pred_stack = stack.predict(data, prediction_length)
        stack_r2_score = r2_score(data['y_test'][-prediction_length:], y_pred_stack)
        stack_mse = mean_squared_error(data['y_test'][-prediction_length:], y_pred_stack)

        # get predictions and scores for bolt
        y_pred_bolt = stack.get_chronos_prediction(torch.tensor(data['y_test'][:-prediction_length]), prediction_length)
        bolt_r2_score = r2_score(data['y_test'][-prediction_length:], y_pred_bolt)
        bolt_mse = mean_squared_error(data['y_test'][-prediction_length:], y_pred_bolt)

        # get predictions and scores for regressor
        y_pred_lgb = stack._lgb_predictions[-prediction_length:]
        lgb_r2_score = r2_score(data['y_test'][-prediction_length:], y_pred_lgb)
        lgb_mse = mean_squared_error(data['y_test'][-prediction_length:], y_pred_lgb)


        # update results dict
        results['dataset'].extend([dataset]*3)
        results['model'].extend(['stack', 'chronos', 'lgb'])
        results['r2'].extend([stack_r2_score, bolt_r2_score, lgb_r2_score])
        results['MSE'].extend([stack_mse, bolt_mse, lgb_mse])
    
    print(pd.DataFrame(results))