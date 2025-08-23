# Multivariate-Time-Series-Anomaly-Detection

This project detects anomalies in multivariate time-series data and highlights which features contributed most to the anomaly. It also provides anomaly scores and interactive plots using Streamlit.

🚀 Features

Train on a “normal” period of data and detect abnormal patterns.

Assign anomaly scores (0–100) to each data point.

Identify top contributing features per anomaly.

Export results as a CSV with anomalies marked.

Interactive visualization of anomalies.

📂 Project Structure
├── app.py                  # Streamlit app for UI
├── anomaly_detection.py    # Core anomaly detection logic
├── sample_data.csv         # Example dataset
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

⚙️ Installation

Clone this repository:

git clone https://github.com/your-username/anomaly-detection.git
cd anomaly-detection


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

▶️ Running the App

Run Streamlit:

streamlit run app.py


Open the local URL in your browser and explore the anomaly detection dashboard.

📊 Output

Interactive plots (highlighting anomalies in red).

Exported CSV with:

anomaly_score

is_anomaly

top_feature

📌 Example

Sample dataset row in exported CSV:

timestamp	sensor1	sensor2	anomaly_score	is_anomaly	top_feature
2023-01-01	10.5	45.2	92.1	1	sensor2
🛠️ Tech Stack

Python

Pandas, NumPy, Scikit-learn (data processing & anomaly detection)

Streamlit (interactive app)

Matplotlib / Plotly (visualization)

📜 License

MIT License
