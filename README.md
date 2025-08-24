# Multivariate Time Series Anomaly Detection

This project implements an AI-based anomaly detection system for multivariate time series data.  
It identifies abnormal patterns across variables, assigns an anomaly score, and highlights the top contributing features.  
The project also includes interactive visualizations using Streamlit.

---

## Live Demo Link

https://multivariate-time-series-anomaly-detection-qpfnvclrtzmzh37qkpy.streamlit.app/

---

## 🚀 Features
- Detect anomalies in multivariate time series data  
- Generate anomaly scores (0–100) for each record  
- Identify **top contributing features** for anomalies  
- Interactive visualizations with Streamlit  
- Export results to CSV for further analysis  

---

## 📂 Project Structure

```text
├── app.py                # Main Streamlit app
├── anomaly_detection.py  # Core logic for anomaly detection
├── data/
│   └── dataset.csv       # Input dataset
├── results/
│   ├── results.csv       # Output file with anomaly scores & top features
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation

```

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## 📊 Usage

Place your input dataset in the data/ folder.

Run the app using the command above.

The app will:

Train on normal patterns

Detect anomalies

Display graphs interactively

Save results in results/results.csv

## 🧠 How It Works

Anomaly Detection: The model learns normal patterns and flags deviations.

Anomaly Score: Calculated as the distance of each data point from the learned normal distribution, scaled to 0–100.

Top Features: Determined using feature importance from the anomaly model (which features contribute most to abnormality).

## 📌 Example Output

Results CSV with anomaly score + top contributing feature(s)

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.

## 📜 License

This project is licensed under the MIT License.
