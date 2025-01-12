import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import sys
import os

# Parse the stock ticker from command line arguments
if len(sys.argv) != 2:
    print("Usage: python multi_feature_model.py <STOCK_TICKER_OR_CSV_FILE>")
    sys.exit(1)

input_arg = sys.argv[1]

# Determine if the input is a ticker or a file path
if input_arg.lower().endswith('.csv'):
    file_path = input_arg
    file_name = os.path.basename(file_path).split('.')[0]
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)
else:
    ticker = input_arg
    file_path = f'data/{ticker.lower()}_stock_data.csv'
    file_name = ticker
    if not os.path.isfile(file_path):
        print(f"Error: The file for ticker '{ticker}' does not exist at '{file_path}'.")
        sys.exit(1)

# Load data
data = pd.read_csv(file_path, skiprows=2)
data.columns = ['Date'] + [f'{col}_{i}' for i, col in enumerate(data.columns[1:], start=1)]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare features and targets
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, :])  # Input features
        Y.append(dataset[i, 0])  # 'Close' price of the first ticker as target
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
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(100, return_sequences=True),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=32, verbose=1)

# Make predictions
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

# Create a new scaler specifically for 'Close' column of the first ticker
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler_close.fit_transform(data.iloc[:, 0].values.reshape(-1, 1))

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

# Calculate rolling standard deviation of the prediction errors
rolling_window = 50
test_errors = Y_test_actual.flatten() - test_predictions.flatten()
rolling_std = pd.Series(test_errors).rolling(window=rolling_window).std().fillna(0).values

# Calculate upper and lower bounds
upper_bound = test_predictions.flatten() + rolling_std
lower_bound = test_predictions.flatten() - rolling_std

# Visualize 
test_index = data.index[train_size + val_size:train_size + val_size + len(Y_test_actual)]

Y_test_actual = Y_test_actual.flatten()
test_predictions = test_predictions.flatten()

plt.figure(figsize=(12, 6))
plt.plot(test_index, Y_test_actual, label="Testing Data Actual")
plt.plot(test_index, test_predictions, label="Testing Data Predictions")
plt.fill_between(test_index, lower_bound, upper_bound, color='gray', alpha=0.2, label="Prediction Range")
plt.legend()
plt.title(f'Stock Price Prediction for {file_name}')  # Dynamic title
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()