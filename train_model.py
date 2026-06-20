import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Data Ingestion & Index Setup
grid_data_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
grid_df = pd.read_csv(grid_data_url, parse_dates=['date'], index_col='date')
grid_df = grid_df.sort_index()

# 2. Feature Engineering (Temporal Indicators & 24-Hour Target Lag)
grid_df['hour']        = grid_df.index.hour
grid_df['dayofweek']   = grid_df.index.dayofweek
grid_df['quarter']     = grid_df.index.quarter
grid_df['month']       = grid_df.index.month
grid_df['dayofyear']   = grid_df.index.dayofyear
grid_df['OT_lag_24h']  = grid_df['OT'].shift(24)

# Drop rows with NaN values introduced by the 24-hour shift
df_clean = grid_df.dropna().copy()

# 3. Feature Selection & Target Definition
features = [
    'hour', 'dayofweek', 'month', 'dayofyear', 'OT_lag_24h',
    'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'
]
X = df_clean[features]
y = df_clean['OT']

# 4. Chronological Train/Test Split (Preserving Time Dependency)
split_index = int(len(df_clean) * 0.80)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 5. Model Training (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("=" * 50)
print(f"Metrics: MAE = {mae:.3f} °C | R² Score = {r2:.4f}")
print("=" * 50)

# 7. Visualization (300-hour Test Window)
plt.figure(figsize=(15, 6))
window = 300

plt.plot(df_clean.index[split_index : split_index + window], y_test.head(window),
         label='Actual Oil Temp', color='#005F73', linewidth=2)
plt.plot(df_clean.index[split_index : split_index + window], predictions[:window],
         label='Predicted Oil Temp', color='#EE9B00', linestyle='--', linewidth=2)

plt.title("Transformer Oil Temperature Prediction - Test Verification", fontsize=14, fontweight='bold')
plt.xlabel("Timeline", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# 8. Model Serialization
model_artifact = {'model': model, 'features': features}
artifact_filename = "transformer_oil_temp_model.joblib"
joblib.dump(model_artifact, artifact_filename)
print(f"Artifact exported successfully: {artifact_filename}\n")

# 9. Simulated Production Inference Engine
loaded_artifact = joblib.load(artifact_filename)
inference_engine = loaded_artifact['model']
model_features = loaded_artifact['features']

# Mock live SCADA single-row data payload
mock_scada_payload = pd.DataFrame([{
    'hour': 14,
    'dayofweek': 2,
    'month': 5,
    'dayofyear': 140,
    'OT_lag_24h': 72.5,
    'HUFL': 8.9, 'HULL': 2.1,
    'MUFL': 14.2, 'MULL': 4.8,
    'LUFL': 22.4, 'LULL': 7.1
}])

# Align incoming SCADA vectors to match exact training feature order
mock_scada_payload = mock_scada_payload[model_features]

# Execute single-row inference
live_prediction = inference_engine.predict(mock_scada_payload)

print("=" * 50)
print(f"Live SCADA Prediction: Expected Oil Temp = {live_prediction[0]:.2f} °C")
print("=" * 50)
