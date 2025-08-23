import streamlit as st
import pandas as pd
from anomaly_detection import detect_anomalies, plot_anomalies

st.set_page_config(page_title="Multivariate Time Series Anomaly Detection", layout="wide")
st.title("🚦 AI-driven Multivariate Time Series Anomaly Detection")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    # read once
    df_full = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df_full.head())

    # Auto-select numeric features but do NOT drop timestamp-like before passing full df
    numeric_features = df_full.select_dtypes(include=["float64","int64"]).columns.tolist()
    # remove timestamp-like columns if they exist in numeric_features
    for col in ["timestamp", "time", "date", "datetime"]:
        if col in numeric_features:
            numeric_features.remove(col)

    st.write("Detected numeric features:", numeric_features)

    # Let user optionally choose timestamp column if any
    ts_candidate = [c for c in df_full.columns if c.lower() in ("timestamp","time","date","datetime")]
    timestamp_col = ts_candidate[0] if ts_candidate else None

    if st.button("🔍 Run Anomaly Detection"):
        with st.spinner("Running anomaly detection..."):
            try:
                # pass full df, plus numeric_features so timestamp is preserved
                result = detect_anomalies(df_full, numeric_features, "results.csv",
                                          contamination=0.05, train_frac=0.7)
                plot_anomalies(result, numeric_features, timestamp_col=timestamp_col,
                               out_plot="anomaly_plot.png", out_feat="feature_importance.png")
                st.success("Anomaly Detection Complete ✅")

                st.write("### Results with Anomaly Scores")
                st.dataframe(result.head(20))

                st.download_button(
                    label="⬇️ Download Full Results CSV",
                    data=open("results.csv","rb").read(),
                    file_name="anomaly_results.csv",
                    mime="text/csv"
                )

                st.image("anomaly_plot.png", caption="Detected Anomalies in Time Series")
                st.image("feature_importance.png", caption="Top Contributing Features")

            except Exception as e:
                st.error(f"Error during detection: {e}")
