import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load dataset
df = pd.read_csv('AAPL_stock_data.csv')

# Remove unwanted rows
df = df.iloc[2:]

# Keep only required columns
df = df[['Open', 'Close']]

# Convert columns to numeric
df['Open'] = pd.to_numeric(df['Open'])
df['Close'] = pd.to_numeric(df['Close'])

# Remove missing values
df.dropna(inplace=True)

# Features and target
X = df[['Open']]
y = df['Close']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Equation of line
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='red')
plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.title("Apple Stock Price Prediction")
plt.show()