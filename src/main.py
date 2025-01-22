import pandas as pd  # requires: pip install pandas
import torch
from chronos.base import BaseChronosPipeline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cpu",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv(
    'datasets/air_passengers.csv'
)


# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
# mean is an fp32 tensor with shape [batch_size, prediction_length]
split = .8
y_true = df["#Passengers"].iloc[int(split*len(df)):]

train = df.iloc[:int(split*len(df))]
quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(train['#Passengers']),
    prediction_length=len(y_true),
    quantile_levels=[0.1, 0.5, 0.9],
)

forecast_index = range(len(train), len(train) + len(y_true))
low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

print(r2_score(y_true, np.array(median)))
plt.figure(figsize=(8, 4))
plt.plot(train["#Passengers"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.plot(forecast_index, y_true, color="green", label="y true")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()


