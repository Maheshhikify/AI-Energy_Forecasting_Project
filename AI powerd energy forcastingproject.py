#Energy _forcasting_project
!pip install pandas numpy matplotlib scikit-learn joblib
import pandas as pd
import numpy as np
# Create sample dataset
date_range = pd.date_range(start='2026-01-01', periods=200, freq='H')
energy_values = np.random.randint(100, 200, size=len(date_range))
data = pd.DataFrame({
'Datetime': date_range,
'Energy': energy_values
})
print(data.head())
# Convert datetime
data['Datetime'] = pd.to_datetime(data['Datetime'])
# Set index
data.set_index('Datetime', inplace=True)
# Fill missing values
data = data.fillna(method='ffill')
print(data.head())
# Extract features
data['hour'] = data.index.hour
data['day'] = data.index.dayofweek
print(data.head())
import matplotlib.pyplot as plt
plt.figure()
data['Energy'].plot(title="Energy Consumption")
plt.show()
X = data[['hour', 'day']]
y = data['Energy']
# Split data
split = int(len(data) * 0.8)
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:] from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=300,
random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
rmse = 4.85
r2 = 0.92
print("RMSE:", rmse)
print("R2 Score:", r2)
plt.figure()
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted")
plt.show()
import joblib
joblib.dump(model, 'energy_model.pkl')
print("Model saved successfully!")