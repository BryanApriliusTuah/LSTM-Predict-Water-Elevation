import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import kagglehub


# import os
# path = kagglehub.dataset_download("lbronchal/venezia")
# files = os.listdir(path)

# csv_file_path = os.path.join(path, 'venezia.csv')
df = pd.read_csv("Dataset BMKG.csv" , sep=";")
print(df.head())

data = df[['Average']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, window_size=7):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 7
X, y = create_sequences(scaled_data, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential()
model.add(LSTM(
    256,
    activation='sigmoid',
    recurrent_activation='tanh',
    input_shape=(X_train.shape[1], 1)
))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae')

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    verbose=1
)

model.save('lstm_model.keras')
joblib.dump(scaler, 'scaler.pkl')

predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual')
plt.plot(predicted, label='Predicted')
plt.title('Water Level Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Water Level (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()