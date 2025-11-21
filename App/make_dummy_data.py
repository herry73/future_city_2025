import pandas as pd
import numpy as np

def make_dummy_data():
    rng = pd.date_range("2024-01-01", periods=24*14, freq="H")  # 14 days hourly

    df = pd.DataFrame({
        "timestamp": rng,
        # Rough “daily pattern” + random noise
        "NO2": 30 + 20*np.sin(2*np.pi * rng.hour / 24) + np.random.randn(len(rng))*5,
        "O3": 40 + 10*np.cos(2*np.pi * rng.hour / 24) + np.random.randn(len(rng))*4,
        "PM10": 25 + 5*np.random.randn(len(rng)),
        "temperature": 10 + 10*np.sin(2*np.pi * (rng.dayofyear) / 365) + np.random.randn(len(rng))*2,
        "humidity": 50 + 20*np.cos(2*np.pi * rng.hour / 24) + np.random.randn(len(rng))*5,
        "noise": 40 + 15*( (rng.hour>=20) | (rng.hour<=1) ) + np.random.randn(len(rng))*3,
    })

    df.to_csv("../data/processed/clean_data_dummy.csv", index=False)
    print("Dummy data written to data/processed/clean_data_dummy.csv")

if __name__ == "__main__":
    make_dummy_data()
