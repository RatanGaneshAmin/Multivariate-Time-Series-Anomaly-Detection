import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)

# Normal data
temperature = 20 + 0.5*np.sin(time/50) + np.random.normal(0,0.2,n_samples)
pressure = 1 + 0.05*np.cos(time/80) + np.random.normal(0,0.01,n_samples)
vibration = 0.05*np.sin(time/25) + np.random.normal(0,0.005,n_samples)

# Inject anomalies after 700
temperature[700:] += np.random.choice([5, -5], size=300) * np.random.rand(300)
pressure[700:] += np.random.choice([0.5, -0.5], size=300) * np.random.rand(300)
vibration[700:] += np.random.choice([0.2, -0.2], size=300) * np.random.rand(300)

df = pd.DataFrame({
    "timestamp": time,
    "temperature": temperature,
    "pressure": pressure,
    "vibration": vibration
})

df.to_csv("multivariate_timeseries.csv", index=False)
print("✅ Dataset saved as multivariate_timeseries.csv")
