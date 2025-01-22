import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1, bidirectional = True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])
    
class LSTMpipeline:

    def __init__(self, lag_size: int = 4, seq_length: int = 4, batch_size: int = 16, num_epochs: int = 200, hidden_size: int = 64, learning_rate: float = .001):
        self._lag_size = lag_size
        self._seq_length = seq_length
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._hidden_size = hidden_size
        self._learning_rate = learning_rate

    def train_model(self, data, features, target, test_size: float = .2, evaluate: bool = True, plot: bool = False):

        self.data = data

        train_loader, test_loader, features = self._preprocess_data(data, features, target, test_size)

        input_size = len(features)

        self.model = LSTM(input_size, self._hidden_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self._learning_rate)
        
        for epoch in range(self._num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f"Epoch {epoch+1}/{self._num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}")
        
        if evaluate:

            self.model.eval()
            y_pred, y_true = [], []

            with torch.no_grad():

                for batch_X, batch_y in test_loader:
                    output = self.model(batch_X)
                    y_pred.extend(output.squeeze().numpy())
                    y_true.extend(batch_y.numpy())

            y_true, y_pred = np.array(y_true), np.array(y_pred)
            y_true = self._y_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred = self._y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            r2 = r2_score(y_true, y_pred)
            print(f"R² Score: {r2:.4f}")
        
            if plot:
                self._plot_results(y_true, y_pred, full_sequence=True)

    def predict(self, data, features, target):

        self.model.eval()
        predictions = list()

        for t in range(1, data.shape[0]+1):

            tmp = pd.concat([self.data, data], ignore_index=True)

            for i in range(self._lag_size):
                tmp[f'lag_{i+1}'] = tmp[target].shift(i+1)

            tmp.dropna(inplace=True)

            X = tmp[features].values
            X_scaled= self._X_scaler.transform(X)

            sequences = np.array(X_scaled[X_scaled.shape[0]-self._seq_length: ])
            sequences = np.expand_dims(sequences, axis=0)

            sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
            with torch.no_grad():
                p = self.model(sequences_tensor).squeeze().numpy()
                p = self._y_scaler.inverse_transform(p.reshape(-1, 1)).flatten().item()
                predictions.append(p)

            if isinstance(data, pd.DataFrame):
                data = data.copy()

            data.loc[t-1, target] = p
            
        return np.array(predictions)

    def _preprocess_data(self, data, features, target, test_size):

        data, features = self._generate_lags(data, features, target)
        X = data[features].values
        y = data[target].values

        X_train, X_test, y_train, y_test = self._train_test_split(X, y, test_size)

        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test)

        train_data = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
        test_data = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test_seq, dtype=torch.float32))

        return DataLoader(train_data, batch_size=self._batch_size, shuffle=False), DataLoader(test_data, batch_size=self._batch_size, shuffle=False), features

    def _generate_lags(self, data, features, target):

        for i in range(self._lag_size):
            data[f'lag_{i+1}'] = data[target].shift(i+1)
            features.append(f'lag_{i+1}')
        data.dropna(inplace=True)

        return data, features
    
    def _train_test_split(self, X, y, test_size: float = .2):

        self._X_scaler = StandardScaler()
        self._y_scaler = StandardScaler()

        idx = int(X.shape[0]*(1-test_size))

        X_train, X_test, y_train, y_test = X[:idx], X[idx:], y[:idx], y[idx:]

        X_train = self._X_scaler.fit_transform(X_train)
        X_test = self._X_scaler.transform(X_test)
        y_train = self._y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = self._y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        return X_train, X_test, y_train, y_test
    
    def _create_sequences(self, X, y):

        sequences, labels = [], []

        for i in range(len(X) - self._seq_length):
            sequences.append(X[i:i+self._seq_length])
            labels.append(y[i+self._seq_length])

        return np.array(sequences), np.array(labels)

    def _plot_results(self, y_true, y_pred, full_sequence=False):
        """
        Plots the true values and predicted values on the same graph.
        Optionally, includes the full sequence of original values along with the predictions.

        :param y_true: True values of the target variable
        :param y_pred: Predicted values
        :param full_sequence: Boolean flag to indicate whether to plot the full original sequence
        """
        plt.figure(figsize=(10, 6))

        # Plot the full data sequence first
        if full_sequence:
            plt.plot(np.arange(len(self.data)), self.data[target].values, label='Full Data', color='gray', alpha=0.5)

        # Plot true values (from the test set)
        plt.plot(np.arange(len(self.data), len(self.data) + len(y_true)), y_true, label='Actual', color='blue')

        # Plot predicted values (from the test set)
        plt.plot(np.arange(len(self.data), len(self.data) + len(y_pred)), y_pred, label='Predicted', color='red')

        plt.legend()
        plt.title('LSTM Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Number of Passengers')
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv('datasets/air_passengers.csv')
    data['Month'] = data['Month'].apply(lambda x: int(x.split('-')[1]))
    eval = data.iloc[int(len(data) * .8):].copy()
    eval.reset_index(drop=True, inplace=True)
    y_true = eval['#Passengers'].values
    eval['#Passengers'] = np.nan
    features = ['Month']
    target = '#Passengers'

    pipeline = LSTMpipeline(lag_size=4, seq_length=12, hidden_size=256, learning_rate=.001, num_epochs=100)
    pipeline.train_model(data, features, target, .2)

    preds = pipeline.predict(eval, features, target)

    # Plot the true vs predicted values after the full data
    pipeline._plot_results(y_true, preds, full_sequence=False)

    # Optionally print R² score
    print(r2_score(y_true, preds))