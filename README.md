Quick start (Streamlit)
Install deps inside the venv if needed: pip install streamlit pandas.
Run the app: streamlit run dashboard.py.
Put your CSV files in data/ or upload one from the sidebar to see an instant preview and charts.
# 🌆 Urban Pulse: Smart City Environmental Decision Engine

A data-driven analytics platform designed for urban planners to monitor environmental stressors and make automated, evidence-based decisions for city improvements.

## 🚀 Key Features
- **Automated Anomaly Detection:** Vectorized algorithms (NumPy/Pandas) to detect "Air-Risk Episodes" and "Night Noise Disturbances" from live sensor streams.
- **Decision Intelligence:** A **Tree Priority Index** algorithm that identifies optimal locations for urban greening by correlating chronic noise pollution with air quality data.
- **Role-Based Analytics:** Custom views for Residents (Health Monitoring), Controllers (Anomaly tracking), and Planners (Future Strategy).
- **Automated ETL Pipeline:** Robust data fetching from **Visual Crossing API** with automated data cleaning and standardization.

## 🛠️ Technical Stack
- **Data Processing:** Python, Pandas (Time-series analysis), NumPy
- **API Integration:** REST APIs (Weather & IoT Sensor data)
- **Visualization:** Plotly Express, Streamlit Custom Layouts
- **Architecture:** Feature-based modular structure (`features.py` logic separation)

## 💡 Practical Impact
The system doesn't just show "that it's noisy"—it calculates **where** the noise is most critical and **recommends** specific interventions, such as where to plant trees to maximize pollution absorption and noise reduction.

---
