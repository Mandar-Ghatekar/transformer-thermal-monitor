---
Development Note: This was my very first project, built using AI Tool to understand basic structural syntax and integration logic. It served as the foundational stepping stone for my understanding of application of data analysis in the field of Power Market Analysis.

## 📊 . The Data Journey: SCADA Sourcing & Analytical Hurdles

### 📡 Data Provenance & The Operational Baseline
The foundational telemetry data used to train and validate this system is sourced from the globally recognized **ETT (Electricity Transformer Temperature) Dataset**. Specifically, the system utilizes the `ETTh1` high-resolution telemetry log, which tracks a multi-phase transformer's operational status over a multi-year window. 

The core target variable I aim to predict is **OT (Oil Temperature)**, which serves as the ultimate indicator of structural thermal stress within the transformer tank.

---

### 🚧 Complex Engineering Challenges Faced

Developing this predictive framework wasn't as simple as passing raw numbers into an AI engine. As an first year electrical student exploring data structures, I hit real-world physical and statistical anomalies that required iterative engineering:

#### 1. The 24-Hour Thermodynamic Lag Problem (`OT_lag_24h`)
* **The Challenge:** Transformers are massive physical assets containing hundreds of liters of insulating oil. Oil has high specific heat capacity; it does not heat up or cool down instantly when electrical load spikes or drops. Early design iterations resulted in poor predictive accuracy because the model didn't understand thermal inertia.
* **The Solution:** a custom 24-hour lagging feature (`df['OT'].shift(24)`) is engineered. By feeding the model the exact core temperature from the previous day at the same hour, we provided a baseline thermal anchor, allowing the system to accurately map active heat accumulation over time.

#### 2. The Seasonal & Diurnal Shift Problem
* **The Challenge:** A transformer is a physical asset operating under two distinct, overlapping cycles: **diurnal** (day/night fluctuations) and **seasonal** (summer/winter macro-climates). 
  
  * **Diurnal Invariance:** At 14:00 (2 PM), solar radiation peaks and industrial demands maximize core heating. At 02:00 (2 AM), ambient cooling drops structural thresholds back to base levels.
  * **Seasonal Variance:** A power transformer operating at maximum capacity under a hot monsoon/summer baseline in July experiences completely different thermal saturation limits and cooling rates than it does during the cold ambient baselines of December. Consumer behavior patterns shift the peak load curves heavily between these months.

* **The Initial Iteration Failure (The 1,000-Hour Constraint Block):**
  During the early proof-of-concept phase of this project, I initially trained the regression model using a sequential slice of only the first 1,000 hours of the dataset to verify code execution. 
  
  While the model achieved decent training metrics on that localized timeline, it **completely failed to predict oil temperatures accurately during the final test hours of the dataset**. Because 1,000 hours represents less than 42 days of continuous telemetry, the training data was completely locked into a single seasonal climate profile. When the model was forced to evaluate test records from later months, it encountered entirely shifted ambient temperatures and seasonal consumer load curves it had never seen before. This resulted in extreme underfitting, massive error spikes, and dangerous false negatives for critical thermal loads.

* **The Solution (Dynamic Rolling-Window Engineering):** To resolve this seasonal blindness without manually scraping external regional weather station databases, we shifted from a static sequential block to an active **Dynamic Rolling-Window Filter** engineered straight from the SCADA time-series axis.

  Now, when an operator queries a specific target timestamp, the backend execution pipeline dynamically slices the entire multi-year database in real time:
  
  1. **Temporal Slicing (D \pm 7):** The algorithm extracts the target day of the year (D) and isolates a tight, localized operational window spanning 7 days before and 7 days after ([D-7, D+7]). This isolates the exact *seasonal* macro-profile (ambient temperature trends and historical consumer behavior patterns for that specific fortnight).
  2. **Diurnal Filtering (H_{target}):** Within that 14-day seasonal slice, the system filters out all records except those matching the exact target hour (H). This isolates the specific *diurnal* load characteristics.
  3. **Feature Aggregation:** The model computes the mean operational vectors (`HUFL`, `MUFL`, `LUFL`, etc.) *only* across this hyper-localized, time-matched historical baseline.

  By feeding the Random Forest engine this localized historical feature intelligence alongside explicit temporal flags (`hour`, `month`, `dayofyear`, `dayofweek`), the system inherently contextualizes shifting seasonal baselines. This evolution from a rigid 1,000-hour block to a dynamic rolling pipeline is a primary driver behind our minimized error rates.
#### 3. Decoding the Phase Loading Telemetry Vector (`HUFL`, `MUFL`, `LUFL`, etc.)
The SCADA dataset contains raw electrical telemetry that directly influences internal winding losses (I^2R copper losses):
* **High Voltage Load (`HUFL` / `HULL`):** High Voltage Useful Load and High Voltage Liquid Load.
* **Medium Voltage Load (`MUFL` / `MULL`):** Medium Voltage Useful/Liquid Load indicators.
* **Low Voltage Load (`LUFL` / `LULL`):** Low Voltage Useful/Liquid Load parameters.

By parsing these six distinct electrical vectors simultaneously, the system learns the non-linear relationship between multi-voltage load distributions and overall core heat dissipation.

---

### 📈 Convergence & Quantitative Performance Optimization

Initially, passing unrefined SCADA variables resulted in high error margins, as the regression model couldn't reconcile sudden load spikes with delayed oil heating. 

By applying data analysis principles—specifically cross-feature correlation indexing, isolating temporal parameters, and embedding the 24-hour thermodynamic lag,it achieved structural convergence. The model's **Mean Absolute Error (MAE)** dropped significantly, ensuring that our real-time temperature forecasts stay tightly aligned with actual physical metrics. This optimization provides utility operators with highly dependable alerts during peak grid stress.
