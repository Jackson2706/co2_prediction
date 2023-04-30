import pandas as pd
import matplotlib.pyplot as plt
import tkinter
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
matplotlib.use("TkAgg")

def create_recursive_data(data, window_size, target_size):
    i = 1
    while i < window_size:
        data[f"co2_{i}"] = data["co2"].shift(-i)
        i = i + 1
    i = 0
    while i < target_size:
        data[f"target_{i+1}"] = data["co2"].shift(-window_size - i)
        i = i + 1   
    data = data.dropna(axis=0)
    return data

data = pd.read_csv("co2.csv")
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("time")
# ax.set_ylabel("CO2")
# plt.show()

window_size = 5
target_size = 3
data = create_recursive_data(data, window_size, target_size)
# print(data)

target = [f"target_{i+1}" for i in range(target_size)]
X = data.drop(target + ["time"], axis=1)
y = data[target]
train_size = 0.8
num_samples = len(X)
x_train = X[:int(num_samples*train_size)]
y_train  = y[:int(num_samples*train_size)]
x_test = X[int(num_samples*train_size):]
y_test = y[int(num_samples*train_size):]


r2 = []
mae = []
mse = []
regs = [LinearRegression() for i in range(target_size)]
for i, reg in enumerate(regs):
    reg.fit(x_train, y_train[f"target_{i+1}"])
    y_predict = reg.predict(x_test)
    r2.append(r2_score(y_true=y_test[f"target_{i+1}"], y_pred=y_predict))
    mae.append(mean_absolute_error(y_true=y_test[f"target_{i+1}"], y_pred=y_predict))
    mse.append(mean_squared_error(y_true=y_test[f"target_{i+1}"], y_pred=y_predict))

# y_predict = reg.predict(x_test)

print(f'R2: {r2}')
print(f'MSE: {mse}')
print(f'MAE: {mae}')

# fig, ax = plt.subplots()

# ax.plot(data["time"][:int(num_samples*train_size)], data["co2"][:int(num_samples*train_size)], label = "train")
# ax.plot(data["time"][int(num_samples*train_size):], data["co2"][int(num_samples*train_size):], label = "test")
# ax.plot(data["time"][int(num_samples*train_size):], y_predict, label = "prediction")
# ax.set_xlabel("Time")
# ax.set_ylabel("Co2")
# ax.legend()
# ax.grid()
# plt.show()