import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(page_title="Mandar Systems - Transformer Analytics", layout="wide")
st.title("⚡ Mandar Systems AI — Transformer Thermal Monitor")
st.markdown("### `System Architecture: Full-Scale Dynamic Rolling Window Extraction`")
st.markdown("---")

@st.cache_resource
def load_full_production_assets():
    loaded_artifact = joblib.load("mandar_systems_transformer_model.joblib")
    grid_data_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(grid_data_url, parse_dates=['date'], index_col='date').sort_index()
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['dayofweek'] = df.index.dayofweek
    df['OT_lag_24h'] = df['OT'].shift(24)
    df = df.dropna()
    return loaded_artifact['model'], loaded_artifact['features'], df

try:
    with st.spinner("⏳ Loading SCADA data..."):
        ai_engine, model_features, historical_db = load_full_production_assets()
    st.sidebar.success("🚀 AI Engine Active")
except Exception as e:
    st.error(f"❌ Failure: {e}")
    st.stop()

st.subheader("Select target evaluation window:")
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("Target Date", datetime.date(2016, 11, 20))
with col2:
    selected_time = st.time_input("Target Time", datetime.time(14, 0))

hour = selected_time.hour
month = selected_date.month
dayofweek = selected_date.weekday()
dayofyear = selected_date.timetuple().tm_yday

window_start = dayofyear - 7
window_end = dayofyear + 7

matched_records = historical_db[
    (historical_db['dayofyear'] >= window_start) &
    (historical_db['dayofyear'] <= window_end) &
    (historical_db['hour'] == hour)
]

if not matched_records.empty:
    profile_sample = matched_records.mean()
    hufl, hull = float(profile_sample['HUFL']), float(profile_sample['HULL'])
    mufl, mull = float(profile_sample['MUFL']), float(profile_sample['MULL'])
    lufl, lull = float(profile_sample['LUFL']), float(profile_sample['LULL'])
    ot_lag_24h = float(profile_sample['OT_lag_24h'])
    st.success(f"📊 Analyzing historical readings for Day {dayofyear}")
else:
    hufl, hull, mufl, mull, lufl, lull, ot_lag_24h = 10.2, 2.5, 12.1, 3.8, 18.4, 5.1, 35.2

user_input_payload = pd.DataFrame([{
    'hour': hour, 'dayofweek': dayofweek, 'month': month, 'dayofyear': dayofyear,
    'OT_lag_24h': ot_lag_24h, 'HUFL': hufl, 'HULL': hull,
    'MUFL': mufl, 'MULL': mull, 'LUFL': lufl, 'LULL': lull
}])[model_features]

live_prediction = ai_engine.predict(user_input_payload)[0]

st.markdown("---")
m_col1, m_col2 = st.columns([1, 2])
with m_col1:
    st.markdown("#### 🔮 Core Temperature Forecast")
    if live_prediction > 45.0:
        st.error(f"⚠️ HIGH THERMAL WORKLOAD: {live_prediction:.2f} °C")
    elif live_prediction > 25.0:
        st.warning(f"⚡ STEADY STATE OPERATIONS: {live_prediction:.2f} °C")
    else:
        st.success(f"✅ COOL STANDBY STATUS: {live_prediction:.2f} °C")

with m_col2:
    st.markdown("#### 📡 Complete Dispatched Feature Vector Array")
    st.dataframe(user_input_payload, hide_index=True)
