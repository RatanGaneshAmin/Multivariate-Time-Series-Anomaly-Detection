import streamlit as st
import pandas as pd
from anomaly_detection import detect_anomalies

st.set_page_config(page_title="Multivariate Time Series Anomaly Detection", layout="wide")
st.title("🚦 AI-driven Multivariate Time Series Anomaly Detection")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df_full = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df_full.head())

    if st.button("🔍 Run Anomaly Detection"):
        with st.spinner("Running anomaly detection..."):
            try:
                # Save uploaded file temporarily
                input_csv = "uploaded.csv"
                with open(input_csv, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run anomaly detection
                result = detect_anomalies(input_csv, "results.csv")
                st.success("Anomaly Detection Complete ✅")

                st.write("### Results with Anomaly Scores")
                st.dataframe(result.head(20))

                st.download_button(
                    label="⬇️ Download Full Results CSV",
                    data=open("results.csv","rb").read(),
                    file_name="anomaly_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error during detection: {e}")
