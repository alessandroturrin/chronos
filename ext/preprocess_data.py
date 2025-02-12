import pandas as pd
import numpy as np

def prepare_data(df: str, target: str, covariates: list[str], train_size: float = .9):

    train = int(len(df)*train_size)

    if len(covariates)>0:
        X_train = df[covariates].iloc[:train].values
        X_test = df[covariates].iloc[train:].values
    else:
        X_train, X_test = None, None

    y_train = np.array(df[target].iloc[:train].values)
    y_test = np.array(df[target].iloc[train:].values)

    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    return data