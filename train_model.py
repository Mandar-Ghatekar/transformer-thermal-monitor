import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Data Ingestion & Sorted Chronological Indexing
grid_data_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
grid_df = pd.read_csv(grid_data_url, parse_dates=['date'], index_col='date')
grid_df = grid_df.sort_index()

# 2. Feature Engineering & 24-Hour Thermodynamic Lag Generation
grid_df['hour']        = grid_df.index.hour
grid_df['dayofweek']   = grid_df.index.dayofweek
grid_df['quarter']     = grid_df.index.quarter
grid_df['month']       = grid_df.index.month
grid_df['dayofyear']   = grid_df.index.dayofyear
grid_df['OT_lag_24h']  = grid_df['OT'].shift(24)

# 3. Purge Null Boundary Rows
master_production_df = grid_df.dropna().copy()

# 4. Feature Space Allocation (Time Matrix + Physical SCADA Current Vectors)
final_features = [
    'hour', 'dayofweek', 'month', 'dayofyear', 'OT_lag_24h',
    'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'
]
X_master = master_production_df[final_features]
y_master = master_production_df['OT']

# 5. Strict Chronological Partitioning (80% Train / 20% Test)
split_index = int(len(master_production_df) * 0.80)
X_train_m, X_test_m = X_master.iloc[:split_index], X_master.iloc[split_index:]
y_train_m, y_test_m = y_master.iloc[:split_index], y_master.iloc[split_index:]

# 6. Model Training (Multi-Core Parallel Execution)
production_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
production_model.fit(X_train_m, y_train_m)

# 7. Model Evaluation & Inference Metrics
production_predictions = production_model.predict(X_test_m)
prod_mae = mean_absolute_error(y_test_m, production_predictions)
prod_r2 = r2_score(y_test_m, production_predictions)

# 8. Definitive Output Performance Report
print("==================================================")
print("     MANDAR SYSTEMS - PRODUCTION REPORT (VERIFIED) ")
print("==================================================")
print(f"🎯 Verified Production MAE     : {prod_mae:.3f} °C")
print(f"📈 Verified Production R² Score : {prod_r2:.4f}")
print("==================================================")

# 9. Chronological Line Overlay Visualization
plt.figure(figsize=(15, 6))
sample_window = 300

plt.plot(master_production_df.index[split_index : split_index + sample_window], y_test_m.head(sample_window),
         label='Actual Oil Temp (SCADA Sensor)', color='#005F73', linewidth=2)

plt.plot(master_production_df.index[split_index : split_index + sample_window], production_predictions[:sample_window],
         label='Predicted Temp (Verified AI)', color='#EE9B00', linestyle='--', linewidth=2)

plt.title("Systems AI - Production Model Verification", fontsize=14, fontweight='bold')
plt.xlabel("Chronological Timeline (Calendar Dates)", fontsize=12)
plt.ylabel("Transformer Oil Temperature (°C)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right', fontsize=11)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

import joblib

# ==============================================================================
# PHASE 12: MODEL SERIALIZATION (SAVING THE ASSET)
# ==============================================================================

# 1. Package the trained model and the exact feature order together in a dictionary
model_artifact = {
    'model': production_model,
    'features': final_features
}

# 2. Export the artifact to a physical file on your disk
artifact_filename = "mandar_systems_transformer_model.joblib"
joblib.dump(model_artifact, artifact_filename)

print("==================================================")
print("     MANDAR SYSTEMS - PRODUCTION ASSET EXPORTED   ")
print("==================================================")
print(f"✅ Model binary securely written to: {artifact_filename}")
print("==================================================\n")


# ==============================================================================
# PHASE 13: PRODUCTION LIVE INFERENCE (LOADING & TESTING)
# ==============================================================================

# 3. Simulate a production environment: Load the model back from the file
loaded_artifact = joblib.load(artifact_filename)
ai_engine = loaded_artifact['model']
model_features = loaded_artifact['features']

print("🚀 Production Asset reloaded successfully into memory.")
print(f"📋 Verified Feature Input Sequence: {model_features}\n")

# 4. Create a mock SCADA sensor data payload for a single hour to test live inference
# Format matches: ['hour', 'dayofweek', 'month', 'dayofyear', 'OT_lag_24h', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
mock_scada_payload = pd.DataFrame([{
    'hour': 14,             # 2:00 PM
    'dayofweek': 2,        # Wednesday
    'month': 5,            # May
    'dayofyear': 140,      # Day 140 of the year
    'OT_lag_24h': 72.5,    # Core temperature yesterday was 72.5°C
    'HUFL': 8.9,           # High voltage phase current load
    'HULL': 2.1,           # High voltage phase light load
    'MUFL': 14.2,          # Medium voltage phase current load
    'MULL': 4.8,           # Medium voltage phase light load
    'LUFL': 22.4,          # Low voltage phase current load
    'LULL': 7.1            # Low voltage phase light load
}])

# Ensure columns match the exact sequence the model was trained on
mock_scada_payload = mock_scada_payload[model_features]

# 5. Execute immediate, single-row inference
live_prediction = ai_engine.predict(mock_scada_payload)

print("==================================================")
print("     LIVE PRODUCTION INFERENCE TEST RESULT        ")
print("==================================================")
print(f"📥 Input SCADA Load State Registered.")
print(f"🔮 Predicted Transformer Oil Temp: {live_prediction[0]:.2f} °C")
print("==================================================")
