import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
matplotlib.use("TkAgg")

def create_recursive_data(data, window_size, future_time):
    i = 1
    while i < window_size:
        data[f"co2_{i}"] = data["co2"].shift(-i)
        i = i + 1

    data["target"] = data["co2"].shift(-i - future_time + 1)
    data = data.dropna(axis=0)
    return data

data = pd.read_csv("co2.csv")
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()


window_size = 5
future__time = 1
data = create_recursive_data(data=data, window_size=window_size, future_time=future__time)
# print(data)

target = "target"
X = data.drop([target,"time"], axis=1)
y = data[target]
train_size = 0.8
num_samples = len(X)
x_train = X[:int(num_samples*train_size)]
y_train  = y[:int(num_samples*train_size)]
x_test = X[int(num_samples*train_size):]
y_test = y[int(num_samples*train_size):]

reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

print(f'R2: {r2_score(y_true=y_test, y_pred=y_predict)}')
print(f'MSE: {mean_squared_error(y_true=y_test, y_pred=y_predict)}')
print(f'MAE: {mean_absolute_error(y_true=y_test, y_pred=y_predict)}')

fig, ax = plt.subplots()

ax.plot(data["time"][:int(num_samples*train_size)], data["co2"][:int(num_samples*train_size)], label = "train")
ax.plot(data["time"][int(num_samples*train_size):], data["co2"][int(num_samples*train_size):], label = "test")
ax.plot(data["time"][int(num_samples*train_size):], y_predict, label = "prediction")
ax.set_xlabel("Time")
ax.set_ylabel("Co2")
ax.legend()
ax.grid()
plt.show()