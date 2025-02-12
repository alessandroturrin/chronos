"""
Module to retrieve datasets for main_meta
"""

import kagglehub

import pandas as pd

from sklearn.preprocessing import LabelEncoder

ALL_DATASETS = ['property_sales', 'netflix', 'weather', 'usd', 'bitcoin']

def get_dataset(dataset):

    if not dataset in ALL_DATASETS:
        raise ValueError(f'Dataset {dataset} not in {ALL_DATASETS}')
    
    if dataset=='property_sales':
        path = kagglehub.dataset_download('htagholdings/property-sales')
        df = pd.read_csv(f'{path}/raw_sales.csv')
        df = df.iloc[:1000]

        # params for dataset
        prediction_length = 14
        target = 'price'
        covariates = ['propertyType', 'postcode', 'bedrooms']
        train_size = .8
        label_encoder = LabelEncoder()

        df['propertyType'] = label_encoder.fit_transform(df['propertyType'])
    
    if dataset=='netflix':
        path = kagglehub.dataset_download("jainilcoder/netflix-stock-price-prediction")
        df = pd.read_csv(f'{path}/NFLX.csv')
        prediction_length = 14
        target = 'Close'
        covariates = ['Open', 'High', 'Low', 'Volume']
        train_size = .8
    
    if dataset=='weather':
        path = kagglehub.dataset_download("gauravsahani/timeseries-analysis-for-whether-dataset")
        df = pd.read_csv(f'{path}/Time-Series Analysis Dataset.csv')
        df = df.dropna()
        prediction_length = 14
        target = 'temperature'
        covariates = ['wind_speed', 'humidity', 'pressure']
        train_size = .8

    if dataset=='usd':
        path = kagglehub.dataset_download("mh0386/usd-to-egp")
        df = pd.read_csv(f'{path}/USD_EGP Historical Data1.csv')
        prediction_length = 14
        target = 'Price'
        covariates = ['Open', 'High', 'Low']
        train_size = .8
        df['Change %'] = df['Change %'].apply(lambda x: float(x.split('%')[0]))
    
    if dataset=='bitcoin':
        path = kagglehub.dataset_download("gallo33henrique/bitcoin-btc-usd-stock-dataset")
        df = pd.read_csv(f'{path}/BTC-USD_stock_data.csv')
        prediction_length = 14
        target = 'Close'
        covariates = ['Open', 'High', 'Low', 'Volume']
        train_size = .8
    
    return df, prediction_length, target, covariates, train_size