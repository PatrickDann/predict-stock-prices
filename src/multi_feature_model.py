import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/apple_stock_data.csv', skiprows=2)
data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'High', 'Low', 'Open', 'Volume']])

# Prepare features and targets
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, :])  # Input features
        Y.append(dataset[i, 0])  # 'Close' price as target
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_dataset(scaled_data, time_step)

# Split into training, validation, and testing sets
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.2)

X_train, Y_train = X[:train_size], Y[:train_size]
X_val, Y_val = X[train_size:train_size + val_size], Y[train_size:train_size + val_size]
X_test, Y_test = X[train_size + val_size:], Y[train_size + val_size:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=32, verbose=1)

# Make predictions
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

# Create a new scaler specifically for 'Close' column
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler_close.fit_transform(data[['Close']])

# Inverse transform predictions and actual values using the new scaler
train_predictions = scaler_close.inverse_transform(train_predictions)
val_predictions = scaler_close.inverse_transform(val_predictions)
test_predictions = scaler_close.inverse_transform(test_predictions)

Y_train_actual = scaler_close.inverse_transform(Y_train.reshape(-1, 1))
Y_val_actual = scaler_close.inverse_transform(Y_val.reshape(-1, 1))
Y_test_actual = scaler_close.inverse_transform(Y_test.reshape(-1, 1))


# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(Y_train_actual, train_predictions))
val_rmse = np.sqrt(mean_squared_error(Y_val_actual, val_predictions))
test_rmse = np.sqrt(mean_squared_error(Y_test_actual, test_predictions))
print(f"Train RMSE: {train_rmse}")
print(f"Validation RMSE: {val_rmse}")
print(f"Test RMSE: {test_rmse}")

# Visualize 
test_index = data.index[train_size + val_size:train_size + val_size + len(Y_test_actual)]

Y_test_actual = Y_test_actual.flatten()
test_predictions = test_predictions.flatten()

plt.figure(figsize=(12, 6))
plt.plot(test_index, Y_test_actual, label="Testing Data Actual")
plt.plot(test_index, test_predictions, label="Testing Data Predictions")
plt.legend()
plt.title("Stock Price Predictions")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

